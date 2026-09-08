# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sub-batch checkpointing: persists a global step's rollouts mid-step."""

from collections.abc import Hashable
import dataclasses
import pickle
import threading
from typing import Any

from absl import logging
import jax
import numpy as np
from orbax.checkpoint import v1 as ocp
from tunix.rl.agentic.agents import agent_types

TrajectoryItem = agent_types.TrajectoryItem

# A global step's snapshots are append-only deltas keyed by
# anchor_step * KEY_BASE + a save counter, where anchor_step is the trainer's
# train_steps when the step started.
KEY_BASE = 1_000_000


@dataclasses.dataclass(frozen=True)
class BatchGeometry:
  """The batch sizes a global step is trained with."""

  full_batch_size: int
  mini_batch_size: int
  # None with sequence packing, where packs replace micro-batches.
  train_micro_batch_size: int | None
  gradient_accumulation_steps: int | None
  num_generations: int
  num_iterations: int


@dataclasses.dataclass(frozen=True)
class SubBatchState:
  """A global step restored from its sub-batch snapshots.

  Attributes:
    global_step: Global step the rollouts belong to.
    anchor_step: Trainer `train_steps` when that global step started.
    geometry: Batch geometry the snapshots were taken under.
    active_groups: The step's rollout groups, in consumption order.
  """

  global_step: int
  anchor_step: int
  geometry: BatchGeometry
  active_groups: list[list[TrajectoryItem]]


class _LatestTwoSteps(ocp.training.preservation_policies.PreservationPolicy):
  """Keeps the snapshots of the two latest global steps.

  A restart can still resume the previous step: its last weight checkpoint may
  be in flight when the next step starts.
  """

  def should_preserve(self, checkpoints, *, context):
    del context
    anchors = sorted({ck.step // KEY_BASE for ck in checkpoints})[-2:]
    return [ck.step // KEY_BASE in anchors for ck in checkpoints]


def _to_numpy(obj: Any) -> Any:
  """Replaces jax arrays with numpy arrays in containers and dataclasses."""
  if isinstance(obj, jax.Array):
    return np.asarray(obj)
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    return dataclasses.replace(
        obj,
        **{
            f.name: _to_numpy(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        },
    )
  if type(obj) is dict:  # pylint: disable=unidiomatic-typecheck
    return {k: _to_numpy(v) for k, v in obj.items()}
  if type(obj) is list:  # pylint: disable=unidiomatic-typecheck
    return [_to_numpy(v) for v in obj]
  if type(obj) is tuple:  # pylint: disable=unidiomatic-typecheck
    return tuple(_to_numpy(v) for v in obj)
  return obj


class SubBatchCheckpointManager:
  """Persists the rollouts of the in-progress global step.

  Snapshots are append-only: each holds the rollout groups and rewards
  recorded since the previous snapshot of the step, and a restore concatenates
  them in order. Training progress is not saved: the trainer checkpoints every
  optimizer update, and a restart retrains the partially accumulated update
  from the restored rollouts with their saved rewards. `record_*` and the
  restored lookups may run on the producer thread; all other methods run on the
  training thread.
  """

  def __init__(self, root_directory: str, *, enable_async: bool = True):
    self._checkpointer = ocp.training.Checkpointer(
        root_directory, preservation_policy=_LatestTwoSteps()
    )
    self._enable_async = enable_async
    self._lock = threading.Lock()
    self._groups: list[list[TrajectoryItem]] = []
    # (prompt_id, group_index, reward) per trajectory.
    self._rewards: list[tuple[Hashable, int, float]] = []
    self._restored_group_ids: frozenset[Hashable] = frozenset()
    self._restored_rewards: dict[tuple[Hashable, int], float] = {}
    self._global_step = 0
    self._anchor_step = 0
    self._geometry: BatchGeometry | None = None
    self._restored_geometry: BatchGeometry | None = None
    self._key = 0
    self._saved_groups = 0
    self._saved_rewards = 0

  @property
  def anchor_step(self) -> int:
    """Trainer `train_steps` when the in-progress global step started."""
    return self._anchor_step

  def has_snapshots(self) -> bool:
    return bool(self._checkpointer.checkpoints)

  def start_step(self, global_step: int, anchor_step: int) -> None:
    """Clears the ledger for a global step starting at trainer `anchor_step`."""
    with self._lock:
      self._groups = []
      self._rewards = []
    self._restored_rewards = {}
    self._global_step = global_step
    self._anchor_step = anchor_step
    self._key = anchor_step * KEY_BASE
    self._saved_groups = 0
    self._saved_rewards = 0

  def record_rollout_group(self, group: list[TrajectoryItem]) -> None:
    """Appends a generated group, in the order it is queued for training."""
    with self._lock:
      self._groups.append(group)

  def record_rewards(
      self, items: list[TrajectoryItem], rewards: np.ndarray
  ) -> None:
    """Appends the rewards computed for training items."""
    with self._lock:
      self._rewards.extend(
          (item.prompt_id, item.group_index, float(reward))
          for item, reward in zip(items, rewards, strict=True)
      )

  def should_skip_group(self, group_id: Hashable) -> bool:
    """Whether the group was restored, so it must not be regenerated."""
    return group_id in self._restored_group_ids

  def restored_rewards(self, items: list[TrajectoryItem]) -> np.ndarray | None:
    """The saved rewards of restored items, or None unless all are saved."""
    keys = [(item.prompt_id, item.group_index) for item in items]
    if not all(key in self._restored_rewards for key in keys):
      return None
    return np.array([self._restored_rewards[key] for key in keys])

  def set_geometry(self, geometry: BatchGeometry) -> None:
    """Records the batch geometry; it must match the restored snapshot's."""
    if (
        self._restored_geometry is not None
        and geometry != self._restored_geometry
    ):
      raise ValueError(
          f"Sub-batch snapshot geometry {self._restored_geometry} does not"
          f" match this run's {geometry}. Relaunch with the original"
          " configuration or a new checkpoint root."
      )
    self._geometry = geometry

  def snapshot(self, *, wait: bool) -> None:
    """Saves what was recorded since the last snapshot; `wait` blocks on it."""
    assert self._geometry is not None, "set_geometry must be called first."
    with self._lock:
      groups = self._groups[self._saved_groups :]
      rewards = self._rewards[self._saved_rewards :]
    if groups or rewards:
      payload = pickle.dumps(
          {
              "global_step": self._global_step,
              "geometry": dataclasses.asdict(self._geometry),
              "groups": [
                  [
                      (
                          t.prompt_id,
                          t.group_index,
                          t.start_step,
                          _to_numpy(t.traj),
                          _to_numpy(t.metadata),
                      )
                      for t in group
                  ]
                  for group in groups
              ],
              "rewards": rewards,
          },
          protocol=pickle.HIGHEST_PROTOCOL,
      )
      checkpointables = {
          "ledger": {"payload": np.frombuffer(payload, dtype=np.uint8)}
      }
      # Deltas must commit in order, and Orbax swaps in a new checkpointer
      # before it waits on the previous save.
      self._checkpointer.wait()
      if self._enable_async:
        self._checkpointer.save_checkpointables_async(
            self._key, checkpointables, force=True
        )
      else:
        self._checkpointer.save_checkpointables(
            self._key, checkpointables, force=True
        )
      self._key += 1
      self._saved_groups += len(groups)
      self._saved_rewards += len(rewards)
    if wait:
      self._checkpointer.wait()

  def restore(self, train_steps: int) -> SubBatchState | None:
    """Loads the latest global step started by `train_steps`.

    Snapshots of later steps followed weights that never became durable and
    are deleted.

    Args:
      train_steps: The trainer's restored apply count.

    Returns:
      The restored state, or None if there is no snapshot.

    Raises:
      ValueError: If every snapshot belongs to a step after `train_steps`.
    """
    keys = sorted(ck.step for ck in self._checkpointer.checkpoints)
    if not keys:
      return None
    anchors = {key // KEY_BASE for key in keys}
    started = [anchor for anchor in anchors if anchor <= train_steps]
    if not started:
      raise ValueError(
          f"Every sub-batch snapshot in {self._checkpointer.directory} belongs"
          f" to a step after trainer step {train_steps}. Restore the matching"
          " trainer checkpoint or use a new checkpoint root."
      )
    anchor_step = max(started)
    for key in keys:
      if key // KEY_BASE > train_steps:
        self._checkpointer.delete(key)
    deltas = [key for key in keys if key // KEY_BASE == anchor_step]
    ledgers = [self._load(key) for key in deltas]
    groups: list[Any] = []
    rewards: dict[tuple[Hashable, int], float] = {}
    for ledger in ledgers:
      groups.extend(ledger["groups"])
      rewards.update(
          ((prompt_id, group_index), reward)
          for prompt_id, group_index, reward in ledger["rewards"]
      )
    state = SubBatchState(
        global_step=ledgers[0]["global_step"],
        anchor_step=anchor_step,
        geometry=BatchGeometry(**ledgers[0]["geometry"]),
        active_groups=[
            [
                TrajectoryItem(
                    prompt_id=prompt_id,
                    group_index=group_index,
                    start_step=start_step,
                    traj=traj,
                    metadata=metadata,
                )
                for prompt_id, group_index, start_step, traj, metadata in group
            ]
            for group in groups
        ],
    )
    self.start_step(state.global_step, state.anchor_step)
    for group in state.active_groups:
      self.record_rollout_group(group)
    self._restored_group_ids = frozenset(
        group[0].prompt_id for group in state.active_groups
    )
    self._restored_rewards = rewards
    self._key = deltas[-1] + 1
    self._saved_groups = len(state.active_groups)
    self._restored_geometry = state.geometry
    logging.info(
        "Sub-batch restore: global step %d with %d groups from %d snapshots.",
        state.global_step,
        len(state.active_groups),
        len(deltas),
    )
    return state

  def _load(self, key: int) -> dict[str, Any]:
    payload = self._checkpointer.load_checkpointables(key)["ledger"]["payload"]
    return pickle.loads(  # pylint: disable=g-unsafe-pickle-load
        np.asarray(payload, dtype=np.uint8).tobytes()
    )

  def wait(self) -> None:
    self._checkpointer.wait()

  def close(self) -> None:
    self._checkpointer.close()
