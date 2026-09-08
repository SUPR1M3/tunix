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

import asyncio
import dataclasses
import itertools
import os
import queue
import threading
from types import SimpleNamespace
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import v1 as ocp
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import sub_batch_checkpoint
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import checkpoint_manager
from tunix.sft import checkpoint_options
from tunix.tests import test_common


class TinyModel(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    del rngs
    self.w = nnx.Param(jnp.zeros((4,)))

  def __call__(self, x):
    return jnp.sum(self.w * x)


class Learner(agentic_rl_learner.AgenticRLLearner):
  """Concrete subclass: stubs the one abstract method so __new__ works."""

  def _process_results(self, *a, **k):
    return []


class FakeCheckpointManager:
  """The trainer checkpoint manager surface the sub-batch hooks use."""

  def __init__(self, latest_step: int | None):
    self.latest = latest_step
    self.saved: list[int] = []
    self.saved_optimizers: list[nnx.Optimizer | None] = []
    self.pinned: list[int | None] = []
    self.loaded: list[int] = []

  def latest_step(self) -> int | None:
    return self.latest

  def has_step(self, step: int) -> bool:
    return self.latest is not None and 0 <= step <= self.latest

  def save(
      self,
      step: int,
      model: nnx.Module | None = None,
      optimizer: nnx.Optimizer | None = None,
      **unused_kwargs,
  ) -> bool:
    del model
    self.saved.append(step)
    self.saved_optimizers.append(optimizer)
    self.latest = step
    return True

  def wait(self) -> None:
    pass

  def pin_step(self, step: int | None) -> None:
    self.pinned.append(step)

  def load_model_params(self, step: int, *unused_args, **unused_kwargs):
    self.loaded.append(step)
    return f"params@{step}"


class FakeEngine(SimpleNamespace):
  """Records weight syncs as (params, increment_global_steps)."""

  def sync_weights(self, params=None, *, increment_global_steps=True):
    self.syncs.append((params, increment_global_steps))


def make_learner(
    *,
    root: str,
    train_steps: int = 0,
    latest_step: int | None = 0,
    global_steps: int = 0,
    k: int = 2,
    num_generations: int = 2,
    num_iterations: int = 1,
    full_batch_size: int = 4,
    process_in_consumer: bool = False,
    packed: bool = False,
) -> Any:
  """Builds a `Learner` without `__init__` around fake collaborators.

  Args:
    root: `training_config.checkpoint_root_directory`.
    train_steps: The trainer's restored apply count.
    latest_step: The trainer's latest checkpoint step, None if none.
    global_steps: The engine's global step.
    k: `gradient_accumulation_steps`.
    num_generations: `algo_config.num_generations`.
    num_iterations: `algo_config.num_iterations`.
    full_batch_size: The learner's resolved full batch size.
    process_in_consumer: Which queue path the learner uses.
    packed: Whether sequence packing is on.

  Returns:
    The learner, with sub-batch checkpointing configured but not initialized.
  """
  learner: Any = Learner.__new__(Learner)
  learner.algo_config = SimpleNamespace(
      sub_batch_checkpointing=True,
      num_generations=num_generations,
      num_iterations=num_iterations,
      off_policy_steps=0,
      use_rollout_logps=True,
      sampler_is=None,
      sampler_rs=None,
      seq_logprob_error_threshold=None,
  )
  learner._training_config = SimpleNamespace(
      get_with_default=lambda key, default: (
          k if key == "gradient_accumulation_steps" else default
      ),
      checkpoint_root_directory=root,
      checkpointing_options=checkpoint_options.TunixCheckpointingOptions(
          save_decision_policy=(
              ocp.training.save_decision_policies.FixedIntervalPolicy(1)
          ),
          enable_async_checkpointing=False,
      ),
  )
  learner._packing_enabled = packed
  learner._process_in_consumer = process_in_consumer
  learner._full_batch_size = full_batch_size
  learner.should_sync_weights = True
  learner._sub_batch = None
  learner._sb_resume = None
  learner._sb_skip_chunks = 0
  learner._sb_skip_updates = 0
  learner._iter_steps = train_steps * k
  model = TinyModel(nnx.Rngs(0))
  learner.rl_engine = FakeEngine(
      global_steps=global_steps,
      syncs=[],
      actor_trainer=SimpleNamespace(
          model=model,
          optimizer=nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param),
          train_steps=train_steps,
          iter_steps=train_steps * k,
          checkpoint_manager=FakeCheckpointManager(latest_step),
      ),
  )
  return learner


def make_group(prompt_id: int, num_generations: int = 2):
  return [
      agent_types.TrajectoryItem(
          prompt_id=prompt_id,
          group_index=i,
          traj={"conversation_tokens": np.array([1, 2, 3]), "reward": 1.0},
          metadata={"generation_id": i},
      )
      for i in range(num_generations)
  ]


# Two updates per epoch: full_batch_size // mini_batch_size.
GEOMETRY = sub_batch_checkpoint.BatchGeometry(
    full_batch_size=4,
    mini_batch_size=2,
    train_micro_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=2,
    num_iterations=1,
)
# Packs replace micro-batches.
PACKED_GEOMETRY = dataclasses.replace(
    GEOMETRY, train_micro_batch_size=None, gradient_accumulation_steps=None
)


def _pack(applies):
  """A packed chunk stand-in: the skip only reads its update flag."""
  return SimpleNamespace(is_update_step=np.array([applies]))


def _plant(
    root,
    *,
    global_step,
    anchor_step,
    prompt_ids,
    num_iterations=1,
    rewards=None,
    packed=False,
):
  manager = sub_batch_checkpoint.SubBatchCheckpointManager(
      os.path.join(root, "sub_batch"), enable_async=False
  )
  manager.set_geometry(
      dataclasses.replace(
          PACKED_GEOMETRY if packed else GEOMETRY,
          num_iterations=num_iterations,
      )
  )
  manager.start_step(global_step, anchor_step)
  for prompt_id in prompt_ids:
    manager.record_rollout_group(make_group(prompt_id))
  for prompt_id, group_rewards in (rewards or {}).items():
    manager.record_rewards(make_group(prompt_id), np.array(group_rewards))
  manager.snapshot(wait=True)
  manager.close()


class SubBatchInitTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = self.create_tempdir().full_path

  def _init(self, **kwargs):
    learner = make_learner(root=self.root, **kwargs)
    learner._init_sub_batch_checkpointing()
    if learner._sub_batch is not None:
      self.addCleanup(learner._sub_batch.close)
      learner._sb_start(learner._sub_batch.anchor_step)
    return learner

  def _trainer_checkpoints(self, learner):
    return learner.rl_engine.actor_trainer.checkpoint_manager

  def test_disabled_is_inert(self):
    learner = make_learner(root=self.root)
    learner.algo_config.sub_batch_checkpointing = False
    learner._init_sub_batch_checkpointing()
    self.assertIsNone(learner._sub_batch)
    self.assertEqual(self._trainer_checkpoints(learner).saved, [])

  def test_fresh_run_checkpoints_and_pins_step_zero_at_start(self):
    learner = make_learner(root=self.root, latest_step=None)
    learner._init_sub_batch_checkpointing()
    self.addCleanup(learner._sub_batch.close)
    # Nothing is written before train() checks the batch geometry.
    self.assertEqual(self._trainer_checkpoints(learner).saved, [])
    learner._sb_start(learner._sub_batch.anchor_step)
    self.assertEqual(self._trainer_checkpoints(learner).saved, [0])
    # Weights only: a relaunch rebuilds the initial optimizer state.
    self.assertEqual(
        self._trainer_checkpoints(learner).saved_optimizers, [None]
    )
    self.assertEqual(self._trainer_checkpoints(learner).pinned, [0])
    self.assertEqual(learner.rl_engine.syncs, [])
    self.assertIsNone(learner._sb_resume)

  def test_fresh_run_with_snapshots_raises(self):
    _plant(self.root, global_step=0, anchor_step=0, prompt_ids=[0])
    with self.assertRaisesRegex(ValueError, "no trainer checkpoint"):
      self._init(latest_step=None)

  def test_mid_step_resume(self):
    _plant(self.root, global_step=4, anchor_step=6, prompt_ids=[16, 17, 18])
    learner = self._init(train_steps=7, latest_step=7, global_steps=5)
    self.assertEqual(learner.rl_engine.global_steps, 4)
    self.assertEqual(learner.policy_version, 4)
    # One update is in the restored weights: its groups are not re-queued.
    self.assertEqual(
        [group[0].prompt_id for group in learner._sb_resume.active_groups],
        [18],
    )
    self.assertEqual(learner._sb_skip_chunks, 0)
    self.assertTrue(learner._sub_batch.should_skip_group(17))
    self.assertEqual(self._trainer_checkpoints(learner).pinned, [6])
    # The rollout engine gets the step's behavior policy, not the restored
    # weights, and global_steps does not move.
    self.assertEqual(learner.rl_engine.syncs, [("params@6", False)])

  def test_resume_in_a_replay_epoch(self):
    _plant(
        self.root,
        global_step=4,
        anchor_step=6,
        prompt_ids=[16, 17, 18, 19],
        num_iterations=2,
    )
    learner = self._init(train_steps=9, latest_step=9, num_iterations=2)
    # Epoch 1 and one replay update are in the restored weights; replay needs
    # every group, so their chunks are skipped instead.
    self.assertLen(learner._sb_resume.active_groups, 4)
    self.assertEqual(learner._sb_skip_chunks, 6)

  def test_packed_resume_in_a_replay_epoch_skips_updates(self):
    _plant(
        self.root,
        global_step=4,
        anchor_step=6,
        prompt_ids=[16, 17, 18, 19],
        num_iterations=2,
        packed=True,
    )
    learner = self._init(
        train_steps=9, latest_step=9, num_iterations=2, packed=True
    )
    # Packs per update vary, so the skip counts the restored updates.
    self.assertEqual(learner._sb_skip_updates, 3)
    self.assertEqual(learner._sb_skip_chunks, 0)

  def test_toggling_packing_on_restart_raises(self):
    _plant(self.root, global_step=4, anchor_step=6, prompt_ids=[16])
    learner = self._init(train_steps=6, latest_step=6, packed=True)
    with self.assertRaisesRegex(ValueError, "does not match"):
      learner._sub_batch.set_geometry(PACKED_GEOMETRY)

  def test_resume_in_a_replay_epoch_needs_every_group(self):
    _plant(
        self.root,
        global_step=4,
        anchor_step=6,
        prompt_ids=[16, 17, 18],
        num_iterations=2,
    )
    with self.assertRaisesRegex(ValueError, "already trained on 4"):
      self._init(train_steps=9, latest_step=9, num_iterations=2)

  def test_resume_at_step_start_syncs_restored_weights(self):
    _plant(self.root, global_step=4, anchor_step=6, prompt_ids=[16])
    learner = self._init(train_steps=6, latest_step=6, global_steps=5)
    self.assertEqual(learner.rl_engine.global_steps, 4)
    self.assertIsNotNone(learner._sb_resume)
    self.assertEqual(learner._sb_skip_chunks, 0)
    self.assertEqual(learner.rl_engine.syncs, [(None, False)])

  def test_finished_step_starts_the_next_step(self):
    _plant(self.root, global_step=4, anchor_step=6, prompt_ids=[16, 17])
    learner = self._init(train_steps=8, latest_step=8, global_steps=4)
    self.assertEqual(learner.rl_engine.global_steps, 5)
    self.assertIsNone(learner._sb_resume)
    self.assertEqual(learner._sb_skip_chunks, 0)
    self.assertEqual(learner._sub_batch.anchor_step, 8)
    self.assertEqual(self._trainer_checkpoints(learner).pinned, [8])

  def test_restart_without_snapshot_continues_from_trainer(self):
    learner = self._init(train_steps=7, latest_step=7, global_steps=5)
    self.assertEqual(learner.rl_engine.global_steps, 5)
    self.assertIsNone(learner._sb_resume)
    self.assertEqual(learner._sub_batch.anchor_step, 7)
    self.assertEqual(learner.rl_engine.syncs, [(None, False)])

  def test_colocated_resume_neither_pins_nor_syncs(self):
    _plant(self.root, global_step=4, anchor_step=6, prompt_ids=[16, 17])
    learner = make_learner(root=self.root, train_steps=7, latest_step=7)
    learner.should_sync_weights = False
    learner._init_sub_batch_checkpointing()
    self.addCleanup(learner._sub_batch.close)
    learner._sb_start(learner._sub_batch.anchor_step)
    self.assertIsNotNone(learner._sb_resume)
    self.assertEqual(self._trainer_checkpoints(learner).pinned, [])
    self.assertEqual(learner.rl_engine.syncs, [])

  def test_snapshot_older_than_trainer_raises(self):
    _plant(self.root, global_step=4, anchor_step=6, prompt_ids=[16])
    with self.assertRaisesRegex(ValueError, "older than trainer step 9"):
      self._init(train_steps=9, latest_step=9)

  def test_snapshot_missing_trained_groups_raises(self):
    _plant(self.root, global_step=4, anchor_step=6, prompt_ids=[16])
    with self.assertRaisesRegex(ValueError, "already trained on 2"):
      self._init(train_steps=7, latest_step=7)


def _set_colocated_recomputed_logps(learner):
  learner.should_sync_weights = False
  learner.algo_config.use_rollout_logps = False


def _set_cadence(interval):
  def mutate(learner):
    learner._training_config.checkpointing_options = (
        checkpoint_options.TunixCheckpointingOptions(
            save_decision_policy=(
                ocp.training.save_decision_policies.FixedIntervalPolicy(
                    interval
                )
            )
        )
    )

  return mutate


class SubBatchConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "critic",
          lambda learner: setattr(learner.rl_engine, "critic_trainer", None),
          "critic",
      ),
      (
          "off_policy",
          lambda learner: setattr(learner.algo_config, "off_policy_steps", 1),
          "off_policy_steps",
      ),
      (
          "no_root",
          lambda learner: setattr(
              learner._training_config, "checkpoint_root_directory", None
          ),
          "checkpoint_root_directory",
      ),
      ("sparse_trainer_checkpoints", _set_cadence(5), "every optimizer update"),
      (
          "colocated_recomputed_logps",
          _set_colocated_recomputed_logps,
          "colocated",
      ),
  )
  def test_rejected(self, mutate, message):
    learner = make_learner(root=self.create_tempdir().full_path)
    mutate(learner)
    with self.assertRaisesRegex(ValueError, message):
      learner._check_sub_batch_config()

  def test_multi_process_rejected(self):
    learner = make_learner(root=self.create_tempdir().full_path)
    with mock.patch.object(jax, "process_count", return_value=2):
      with self.assertRaisesRegex(ValueError, "single-process"):
        learner._check_sub_batch_config()


class SubBatchTrainLoopHelpersTest(absltest.TestCase):

  def test_untrained_drops_leading_chunks_across_calls(self):
    learner = make_learner(root=self.create_tempdir().full_path)
    learner._sb_skip_chunks = 5
    learner._iter_steps = 7
    self.assertEqual(learner._sb_untrained([1, 2, 3]), [])
    self.assertEqual(learner._sb_untrained([4, 5, 6]), [6])
    self.assertEqual(learner._sb_untrained([7]), [7])
    # Skipped chunks are not trained, so `_iter_steps` stops counting them.
    self.assertEqual(learner._iter_steps, 2)

  def test_ends_update(self):
    learner = make_learner(root=self.create_tempdir().full_path, k=2)
    trainer = learner.rl_engine.actor_trainer
    trainer.iter_steps = 4
    self.assertFalse(learner._sb_ends_update([1]))
    self.assertTrue(learner._sb_ends_update([1, 2]))
    trainer.iter_steps = 5
    self.assertTrue(learner._sb_ends_update([1]))

  def test_packed_skip_counts_update_flags(self):
    learner = make_learner(root=self.create_tempdir().full_path)
    learner._packing_enabled = True
    learner._sb_skip_updates = 2
    learner._iter_steps = 6
    packs = [_pack(applies) for applies in (False, True, False, True, True)]
    self.assertEqual(learner._sb_untrained(packs[:3]), [])
    self.assertEqual(learner._sb_untrained(packs[3:]), packs[4:])
    self.assertEqual(learner._iter_steps, 2)

  def test_packed_ends_update_reads_the_flag(self):
    learner = make_learner(root=self.create_tempdir().full_path)
    learner._packing_enabled = True
    self.assertFalse(learner._sb_ends_update([_pack(False)]))
    self.assertTrue(learner._sb_ends_update([_pack(True)]))


class SubBatchRewardsTest(absltest.TestCase):

  def test_restored_groups_reuse_saved_rewards(self):
    root = self.create_tempdir().full_path
    _plant(
        root,
        global_step=2,
        anchor_step=4,
        prompt_ids=[8, 9],
        rewards={8: [1.0, 0.5]},
    )
    learner = make_learner(root=root, train_steps=4, latest_step=4)
    learner._init_sub_batch_checkpointing()
    self.addCleanup(learner._sub_batch.close)
    learner.reward_manager = mock.Mock(
        return_value={"rewards": np.array([3.0, 4.0]), "log_metrics": {}}
    )
    learner.rl_engine.buffer_metrics_async = mock.Mock()
    learner._rewards_window_lock = threading.Lock()
    learner._train_rewards_window = []
    learner._eval_rewards_window = []
    scored, unscored = learner._sb_resume.active_groups

    def rewards(items, mode=rl_engine_lib.Mode.TRAIN):
      return learner._compute_rewards(
          prompts=["p", "p"],
          completions=["c", "c"],
          mode=mode,
          trajectory_items=items,
      )

    np.testing.assert_array_equal(rewards(scored), [1.0, 0.5])
    learner.reward_manager.assert_not_called()
    np.testing.assert_array_equal(rewards(unscored), [3.0, 4.0])
    # Eval rewards are never reused or saved.
    np.testing.assert_array_equal(
        rewards(scored, rl_engine_lib.Mode.EVAL), [3.0, 4.0]
    )
    learner._sub_batch.set_geometry(GEOMETRY)
    learner._sub_batch.snapshot(wait=True)
    manager = sub_batch_checkpoint.SubBatchCheckpointManager(
        os.path.join(root, "sub_batch"), enable_async=False
    )
    self.addCleanup(manager.close)
    manager.restore(train_steps=4)
    np.testing.assert_array_equal(manager.restored_rewards(scored), [1.0, 0.5])
    np.testing.assert_array_equal(
        manager.restored_rewards(unscored), [3.0, 4.0]
    )


def _arrays(state) -> dict[str, np.ndarray]:
  """An nnx state's arrays, by path."""
  return {
      jax.tree_util.keystr(path): np.asarray(leaf)
      for path, leaf in jax.tree_util.tree_leaves_with_path(state)
  }


class SubBatchBehaviorPolicyTest(parameterized.TestCase):
  """Loads the step's behavior policy from real trainer checkpoints."""

  def _learner(self, lora=False):
    """A learner at trainer step 2, with a real model and checkpoints."""
    root = self.create_tempdir().full_path
    learner = make_learner(root=root, train_steps=2, latest_step=2)
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(), rngs=nnx.Rngs(0)
    )
    trainer = learner.rl_engine.actor_trainer
    trainer.model = test_common.get_lora_model(model) if lora else model
    trainer.checkpoint_manager = checkpoint_manager.CheckpointManager(root)
    self.addCleanup(trainer.checkpoint_manager.close)
    return learner

  @parameterized.parameters(False, True)
  def test_mid_step_resume_syncs_the_step_start_weights(self, lora):
    learner = self._learner(lora)
    trainer = learner.rl_engine.actor_trainer
    trainer.checkpoint_manager.save(
        0, trainer.model, save_only_lora_params=lora, force=True
    )
    trainer.checkpoint_manager.wait()
    step_start = _arrays(nnx.state(trainer.model))
    # The step's updates; LoRA trains only the adapters.
    trained = nnx.state(trainer.model, nnx.LoRAParam if lora else nnx.Param)
    nnx.update(trainer.model, jax.tree.map(lambda x: x + 1.0, trained))

    learner._sb_sync_rollout_weights(0)

    [(params, increment_global_steps)] = learner.rl_engine.syncs
    self.assertFalse(increment_global_steps)
    # Every weight is the step's start: under LoRA, the restored adapters on
    # the live base.
    synced = _arrays(params)
    self.assertEqual(synced.keys(), step_start.keys())
    for path, value in synced.items():
      np.testing.assert_array_equal(value, step_start[path], err_msg=path)
    self.assertTrue(
        all(
            leaf.sharding.memory_kind == "pinned_host"
            for leaf in jax.tree.leaves(params)
        )
    )

  def test_missing_step_start_checkpoint_raises(self):
    learner = self._learner()
    with self.assertRaisesRegex(RuntimeError, "Checkpoint 0"):
      learner._sb_sync_rollout_weights(0)
    self.assertEqual(learner.rl_engine.syncs, [])


class _RecordingOrchestrator:
  """Records the group ids the producer streams agent-env pairs for."""

  def __init__(self):
    self.group_ids = []
    self._done = asyncio.Event()

  async def run_producers_from_stream(self, pairs_stream, **unused_kwargs):
    async for _, env in pairs_stream:
      self.group_ids.append(env.extra_kwargs["group_id"])
    self._done.set()

  async def yield_batches(self, batch_size):
    del batch_size
    await self._done.wait()
    return
    yield  # pylint: disable=unreachable


class SubBatchProducerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = self.create_tempdir().full_path

  def test_skipped_groups_are_not_generated(self):
    learner = make_learner(root=self.root, global_steps=5, full_batch_size=4)
    learner._background_tasks = set()
    learner._create_agent_env_pair = lambda example, group_id, pair_index: (
        None,
        SimpleNamespace(extra_kwargs={"group_id": group_id}),
    )

    async def run():
      learner.loop = asyncio.get_running_loop()
      orchestrator = _RecordingOrchestrator()
      async for _ in learner._orchestrator_producer(
          orchestrator,
          [{}, {}, {}],
          skip_group=lambda group_id: group_id == 9,
          first_group_id=8,
      ):
        pass
      return orchestrator.group_ids

    self.assertEqual(asyncio.run(run()), [8, 10])

  def test_restored_groups_queue_first_and_new_groups_are_recorded(self):
    _plant(self.root, global_step=2, anchor_step=4, prompt_ids=[8, 9])
    learner = make_learner(root=self.root, global_steps=2)
    manager = sub_batch_checkpoint.SubBatchCheckpointManager(
        os.path.join(self.root, "sub_batch"), enable_async=False
    )
    self.addCleanup(manager.close)
    learner._sub_batch = manager
    learner._sb_resume = manager.restore(train_steps=4)
    learner._batch_to_train_example = lambda batch_results, mode: [
        batch_results[0].prompt_id
    ]

    async def orchestrator_producer(skip_group, first_group_id, **unused):
      for group_id in range(first_group_id, first_group_id + 3):
        if not skip_group(group_id):
          yield make_group(group_id)

    learner._orchestrator_producer = orchestrator_producer
    data = queue_lib.SimpleDataQueue(maxsize=0)
    asyncio.run(learner._producer(None, queue.Queue(), data))
    self.assertEqual(list(iter(data.get, None)), [8, 9, 10])
    manager.set_geometry(GEOMETRY)
    manager.snapshot(wait=True)
    self.assertEqual(
        [group[0].prompt_id for group in manager.restore(4).active_groups],
        [8, 9, 10],
    )


class _TrainEngine(FakeEngine):
  """Logs the trained chunks; the trainer applies every `k` of them.

  A pack instead applies when its update flag is set, and is logged with its
  groups and that flag.
  """

  def update_actor(self, chunks, eval_dataset, skip_jit):
    del skip_jit
    trainer = self.actor_trainer
    for chunk in chunks:
      trainer.iter_steps += 1
      if chunk.segment_ids is None:
        group_id = int(chunk.prompt_ids[0, 0])
        self.log.append(("train", group_id))
        self.trained_rewards.append((group_id, float(chunk.advantages[0, 0])))
        if eval_dataset:
          self.evals.append(group_id)
        applies = trainer.iter_steps % self.k == 0
      else:
        applies = bool(np.asarray(chunk.is_update_step).item())
        # Every token of a packed group is its id + 1; padding is 0.
        tokens = np.unique(np.asarray(chunk.completion_ids))
        groups = tuple(int(token) - 1 for token in tokens if token)
        self.log.append(("train", groups, applies))
      if applies:
        trainer.train_steps += 1
        trainer.checkpoint_manager.save(trainer.train_steps)

  def sync_weights(self, params=None, *, increment_global_steps=True):
    super().sync_weights(params, increment_global_steps=increment_global_steps)
    if increment_global_steps:
      self.global_steps += 1

  def buffer_metrics(self, *unused_args, **unused_kwargs):
    pass

  buffer_metrics_async = buffer_metrics
  close = buffer_metrics


async def _orchestrator_producer(
    orchestrator,
    prompt_iterator,
    num_generations=1,
    collect_mode="Token",
    skip_group=None,
    first_group_id=None,
):
  """Yields one group per prompt, numbered like the real producer."""
  del orchestrator, num_generations, collect_mode
  if isinstance(prompt_iterator, list):  # Eval prompts.
    for _ in prompt_iterator:
      yield make_group(-1)
    return
  group_id = first_group_id
  async for _ in prompt_iterator:
    if not skip_group(group_id):
      yield make_group(group_id)
    group_id += 1


def _train_examples(batch_results, rewards):
  rows = len(batch_results)
  return [
      agentic_rl_learner.TrainExample(
          prompt_ids=np.array([[item.prompt_id] for item in batch_results]),
          prompt_mask=np.ones((rows, 1)),
          completion_ids=np.zeros((rows, 1)),
          completion_mask=np.ones((rows, 1)),
          # Rewards stand in for advantages, so tests see what each chunk used.
          advantages=np.asarray(rewards, dtype=np.float32)[:, None],
          ref_per_token_logps=None,
          old_per_token_logps=None,
      )
  ]


def _packable_train_examples(batch_results, rewards):
  """Rows of 2 tokens, 4 for groups 0, 4, ...; group g's tokens are g + 1."""
  rows = len(batch_results)
  completion_ids = np.zeros((rows, 3), dtype=np.int32)
  completion_mask = np.zeros((rows, 3), dtype=np.float32)
  for row, item in enumerate(batch_results):
    length = 3 if item.prompt_id % 4 == 0 else 1
    completion_ids[row, :length] = item.prompt_id + 1
    completion_mask[row, :length] = 1
  return [
      agentic_rl_learner.TrainExample(
          prompt_ids=np.array([[item.prompt_id + 1] for item in batch_results]),
          prompt_mask=np.ones((rows, 1)),
          completion_ids=completion_ids,
          completion_mask=completion_mask,
          advantages=np.asarray(rewards, dtype=np.float32),
          ref_per_token_logps=None,
          old_per_token_logps=None,
      )
  ]


_SNAPSHOT = sub_batch_checkpoint.SubBatchCheckpointManager.snapshot


class SubBatchTrainTest(parameterized.TestCase):
  """Runs `train()`: 4 groups per step, an update every 2 of them."""

  def setUp(self):
    super().setUp()
    self.root = self.create_tempdir().full_path
    # Every reward_manager call scores anew, as a stochastic judge would.
    self._rewards = itertools.count()

  def _train(
      self,
      *,
      max_steps,
      eval_dataset=None,
      k=2,
      process_in_consumer=False,
      packed=False,
      **learner_kwargs,
  ):
    learner = make_learner(root=self.root, k=k, packed=packed, **learner_kwargs)
    learner.rl_engine = _TrainEngine(
        **vars(learner.rl_engine),
        k=k,
        log=[],
        trained_rewards=[],
        evals=[],
        perf_v2=mock.MagicMock(),
        cluster_config=SimpleNamespace(
            training_config=SimpleNamespace(eval_every_n_steps=1),
            # One packed row per pack.
            role_to_mesh={
                rl_engine_lib.Role.ACTOR: SimpleNamespace(shape={"fsdp": 1})
            },
        ),
    )
    micro = 2 // k
    learner._training_config.mini_batch_size = 2
    learner._training_config.train_micro_batch_size = micro
    learner._training_config.max_steps = max_steps
    # Packs hold up to 8 tokens.
    learner._training_config.max_seq_token_per_tpu = 8
    # `train()` processes groups in the consumer when this equals `micro`.
    learner._compute_logps_micro_batch_size = (
        micro if process_in_consumer else 1
    )
    learner._rollout_sync_lock = agentic_utils.RolloutSyncLock()
    learner._iter_steps = learner.rl_engine.actor_trainer.iter_steps
    learner._eval_iter_steps = 0
    learner._last_eval_train_step = -1
    learner._train_rewards_window = []
    learner._eval_rewards_window = []
    learner._rewards_window_lock = threading.Lock()
    learner._global_step_start_time = 0.0
    learner.policy_version = 0
    learner._build_orchestrator = lambda: None
    learner._orchestrator_producer = _orchestrator_producer
    learner.reward_manager = mock.Mock(
        side_effect=lambda prompts, completions, **unused: {
            "rewards": np.full(len(prompts), float(next(self._rewards))),
            "log_metrics": {},
        }
    )

    def batch_to_train_example(batch_results, mode):
      rewards = learner._compute_rewards(
          prompts=["p"] * len(batch_results),
          completions=["c"] * len(batch_results),
          mode=mode,
          trajectory_items=batch_results,
      )
      if packed:
        return _packable_train_examples(batch_results, rewards)
      return _train_examples(batch_results, rewards)

    learner._batch_to_train_example = mock.Mock(
        side_effect=batch_to_train_example
    )
    log = learner.rl_engine.log

    def logged_snapshot(manager, *, wait):
      log.append(("snapshot", wait))
      _SNAPSHOT(manager, wait=wait)

    self.enter_context(
        mock.patch.object(
            sub_batch_checkpoint.SubBatchCheckpointManager,
            "snapshot",
            logged_snapshot,
        )
    )
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    self.addCleanup(loop.close)
    self.addCleanup(thread.join)
    self.addCleanup(loop.call_soon_threadsafe, loop.stop)
    learner.loop = loop
    learner.train(
        [{"prompts": ["p"] * 4} for _ in range(max_steps + 1)], eval_dataset
    )
    return learner

  def _checkpoints(self, learner):
    return learner.rl_engine.actor_trainer.checkpoint_manager

  def test_rollouts_are_durable_before_each_update(self):
    learner = self._train(max_steps=1, latest_step=None, num_iterations=2)
    self.assertEqual(
        learner.rl_engine.log,
        [
            ("snapshot", False),
            ("train", 0),
            ("snapshot", True),
            ("train", 1),
            ("snapshot", False),
            ("train", 2),
            ("snapshot", True),
            ("train", 3),
            # The replay epoch takes no snapshot.
            ("train", 0),
            ("train", 1),
            ("train", 2),
            ("train", 3),
        ],
    )
    # The next step starts at, and pins, the step's final weights.
    self.assertEqual(learner._sub_batch.anchor_step, 4)
    self.assertEqual(self._checkpoints(learner).pinned, [0, 4])
    self.assertEqual(learner.rl_engine.global_steps, 1)

  def test_resume_skips_the_updates_the_restored_weights_hold(self):
    _plant(
        self.root,
        global_step=0,
        anchor_step=0,
        prompt_ids=[0, 1, 2, 3],
        num_iterations=2,
    )
    learner = self._train(
        max_steps=1,
        train_steps=3,
        latest_step=3,
        global_steps=1,
        num_iterations=2,
    )
    # Epoch 1 and the first replay update are skipped, and no group is
    # regenerated.
    self.assertEqual(learner.rl_engine.log, [("train", 2), ("train", 3)])
    trainer = learner.rl_engine.actor_trainer
    self.assertEqual(trainer.train_steps, 4)
    self.assertEqual(learner._iter_steps, trainer.iter_steps)
    self.assertIsNone(learner._sb_resume)
    self.assertEqual(self._checkpoints(learner).pinned, [0, 4])
    self.assertEqual(
        learner.rl_engine.syncs, [("params@0", False), (None, True)]
    )

  def test_single_epoch_resume_does_not_requeue_trained_groups(self):
    _plant(self.root, global_step=0, anchor_step=0, prompt_ids=[0, 1, 2])
    learner = self._train(max_steps=1, train_steps=1, latest_step=1)
    # Groups 0 and 1 are in the restored update: only the restored group 2
    # and the new group 3 are processed and trained, and the step ends after
    # the one remaining update.
    processed = [
        call.kwargs["batch_results"][0].prompt_id
        for call in learner._batch_to_train_example.call_args_list
    ]
    self.assertEqual(processed, [2, 3])
    self.assertEqual(
        learner.rl_engine.log,
        [
            ("snapshot", False),
            ("train", 2),
            ("snapshot", True),
            ("train", 3),
        ],
    )
    trainer = learner.rl_engine.actor_trainer
    self.assertEqual(trainer.train_steps, 2)
    self.assertEqual(learner._iter_steps, trainer.iter_steps)
    self.assertEqual(learner.rl_engine.global_steps, 1)
    self.assertEqual(self._checkpoints(learner).pinned, [0, 2])

  @parameterized.parameters(False, True)
  def test_restart_reuses_the_rewards_the_step_trained_with(
      self, process_in_consumer
  ):
    kwargs = dict(
        max_steps=1,
        num_iterations=2,
        k=1 if process_in_consumer else 2,
        process_in_consumer=process_in_consumer,
    )
    first = self._train(latest_step=None, **kwargs)
    # Restart as if the step's last trainer checkpoint never became durable:
    # the last replay update is retrained from the restored rollouts.
    second = self._train(train_steps=3, latest_step=3, **kwargs)
    second.reward_manager.assert_not_called()
    first_rewards = dict(first.rl_engine.trained_rewards)
    self.assertNotEmpty(second.rl_engine.trained_rewards)
    for group_id, reward in second.rl_engine.trained_rewards:
      self.assertEqual(reward, first_rewards[group_id])

  def test_resumed_step_does_not_repeat_eval(self):
    _plant(self.root, global_step=0, anchor_step=0, prompt_ids=[0, 1])
    learner = self._train(
        max_steps=2,
        latest_step=0,
        global_steps=1,
        eval_dataset=[{"prompts": ["e"]}],
    )
    self.assertEqual(
        [entry for entry in learner.rl_engine.log if entry[0] == "train"],
        [("train", group_id) for group_id in range(8)],
    )
    self.assertEqual(learner.rl_engine.evals, [4])

  def test_packed_rollouts_are_durable_before_each_update(self):
    learner = self._train(
        max_steps=1, latest_step=None, num_iterations=2, packed=True
    )
    # Groups 0 and 1 take two packs, groups 2 and 3 one.
    self.assertEqual(
        learner.rl_engine.log,
        [
            ("snapshot", False),
            ("train", (0,), False),
            ("snapshot", True),
            ("train", (1,), True),
            ("snapshot", True),
            ("train", (2, 3), True),
            ("train", (0,), False),
            ("train", (1,), True),
            ("train", (2, 3), True),
        ],
    )

  def test_packed_resume_skips_the_packs_of_restored_updates(self):
    kwargs = dict(max_steps=1, num_iterations=2, packed=True)
    self._train(latest_step=None, **kwargs)
    # Restart as if the step's last update never became durable.
    learner = self._train(train_steps=3, latest_step=3, **kwargs)
    self.assertEqual(learner.rl_engine.log, [("train", (2, 3), True)])
    trainer = learner.rl_engine.actor_trainer
    self.assertEqual(trainer.train_steps, 4)
    self.assertEqual(learner._iter_steps, trainer.iter_steps)

  def test_packed_single_epoch_resume_does_not_requeue_trained_groups(self):
    self._train(max_steps=1, latest_step=None, packed=True)
    learner = self._train(
        max_steps=1, train_steps=1, latest_step=1, packed=True
    )
    self.assertEqual(
        learner.rl_engine.log, [("snapshot", True), ("train", (2, 3), True)]
    )
    self.assertEqual(learner.rl_engine.actor_trainer.train_steps, 2)


if __name__ == "__main__":
  absltest.main()
