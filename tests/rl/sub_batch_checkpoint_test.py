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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.rl import sub_batch_checkpoint
from tunix.rl.agentic.agents import agent_types

KEY_BASE = sub_batch_checkpoint.KEY_BASE
GEOMETRY = sub_batch_checkpoint.BatchGeometry(
    full_batch_size=4,
    mini_batch_size=2,
    train_micro_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=2,
    num_iterations=1,
)


def _group(prompt_id, num_generations=2):
  return [
      agent_types.TrajectoryItem(
          prompt_id=prompt_id,
          group_index=i,
          traj={
              "conversation_tokens": np.array([1, 2, prompt_id]),
              "old_logps": jnp.array([0.5, 0.25]),
              "reward": float(i),
          },
          metadata={"generation_id": i},
      )
      for i in range(num_generations)
  ]


class SubBatchCheckpointManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.root = self.create_tempdir().full_path

  def _manager(self, enable_async=False):
    manager = sub_batch_checkpoint.SubBatchCheckpointManager(
        self.root, enable_async=enable_async
    )
    self.addCleanup(manager.close)
    manager.set_geometry(GEOMETRY)
    return manager

  def _keys(self, manager):
    manager.wait()
    return sorted(ck.step for ck in manager._checkpointer.checkpoints)

  def _write(self, global_step, anchor_step, prompt_ids):
    manager = sub_batch_checkpoint.SubBatchCheckpointManager(
        self.root, enable_async=False
    )
    manager.set_geometry(GEOMETRY)
    manager.start_step(global_step, anchor_step)
    for prompt_id in prompt_ids:
      manager.record_rollout_group(_group(prompt_id))
      manager.snapshot(wait=False)
    manager.close()

  @parameterized.parameters(False, True)
  def test_restore_round_trips_groups(self, enable_async):
    manager = self._manager(enable_async)
    manager.start_step(global_step=3, anchor_step=6)
    manager.record_rollout_group(_group(12))
    manager.record_rollout_group(_group(13))
    manager.snapshot(wait=True)

    state = self._manager().restore(train_steps=7)

    self.assertEqual(state.global_step, 3)
    self.assertEqual(state.anchor_step, 6)
    self.assertEqual(state.geometry, GEOMETRY)
    self.assertEqual(
        [
            [(t.prompt_id, t.group_index) for t in group]
            for group in state.active_groups
        ],
        [[(12, 0), (12, 1)], [(13, 0), (13, 1)]],
    )
    item = state.active_groups[1][1]
    self.assertEqual(item.metadata, {"generation_id": 1})
    self.assertEqual(item.traj["reward"], 1.0)
    np.testing.assert_array_equal(item.traj["conversation_tokens"], [1, 2, 13])
    self.assertIsInstance(item.traj["old_logps"], np.ndarray)
    np.testing.assert_array_equal(item.traj["old_logps"], [0.5, 0.25])

  def test_snapshots_hold_only_new_groups(self):
    manager = self._manager()
    manager.start_step(global_step=2, anchor_step=4)
    manager.record_rollout_group(_group(8))
    manager.snapshot(wait=False)
    manager.snapshot(wait=True)
    self.assertEqual(self._keys(manager), [4 * KEY_BASE])
    manager.record_rollout_group(_group(9))
    manager.record_rollout_group(_group(10))
    manager.snapshot(wait=True)
    self.assertEqual(self._keys(manager), [4 * KEY_BASE, 4 * KEY_BASE + 1])
    delta = manager._load(4 * KEY_BASE + 1)
    self.assertEqual([group[0][0] for group in delta["groups"]], [9, 10])
    state = self._manager().restore(train_steps=4)
    self.assertEqual(
        [group[0].prompt_id for group in state.active_groups], [8, 9, 10]
    )

  def test_restores_saved_rewards(self):
    manager = self._manager()
    manager.start_step(global_step=0, anchor_step=0)
    manager.record_rollout_group(_group(0))
    manager.record_rollout_group(_group(1))
    manager.snapshot(wait=True)
    # Rewards computed after their group was saved get a snapshot of their own.
    manager.record_rewards(_group(0), np.array([1.0, 0.5]))
    manager.snapshot(wait=True)
    self.assertEqual(self._keys(manager), [0, 1])

    restored = self._manager()
    state = restored.restore(train_steps=0)

    np.testing.assert_array_equal(
        restored.restored_rewards(state.active_groups[0]), [1.0, 0.5]
    )
    self.assertIsNone(restored.restored_rewards(state.active_groups[1]))
    # After another restart, the latest saved rewards of a group win.
    restored.record_rewards(_group(0), np.array([2.0, 2.5]))
    restored.snapshot(wait=True)
    again = self._manager()
    state = again.restore(train_steps=0)
    np.testing.assert_array_equal(
        again.restored_rewards(state.active_groups[0]), [2.0, 2.5]
    )

  def test_retention_keeps_the_two_latest_steps(self):
    manager = self._manager()
    for global_step in (1, 2, 3):
      manager.start_step(global_step, anchor_step=2 * global_step)
      for prompt_id in range(2):
        manager.record_rollout_group(_group(prompt_id))
        manager.snapshot(wait=False)
    self.assertEqual(
        self._keys(manager),
        [4 * KEY_BASE, 4 * KEY_BASE + 1, 6 * KEY_BASE, 6 * KEY_BASE + 1],
    )

  def test_snapshot_converts_nested_jax_arrays(self):
    manager = self._manager()
    manager.start_step(global_step=0, anchor_step=0)
    traj = agent_types.Trajectory(
        steps=[agent_types.Step(info={"logprobs": jnp.array([-0.5])})]
    )
    manager.record_rollout_group([
        agent_types.TrajectoryItem(
            prompt_id=0,
            traj=traj,
            metadata={"turns": {0: jnp.array([1]), "count": 1}},
        )
    ])
    manager.snapshot(wait=True)

    item = self._manager().restore(train_steps=0).active_groups[0][0]

    self.assertIsInstance(item.traj.steps[0].info["logprobs"], np.ndarray)
    self.assertIsInstance(item.metadata["turns"][0], np.ndarray)
    self.assertEqual(item.metadata["turns"]["count"], 1)

  def test_restore_without_snapshots_returns_none(self):
    manager = self._manager()
    self.assertFalse(manager.has_snapshots())
    self.assertIsNone(manager.restore(train_steps=5))

  def test_restore_skips_and_deletes_snapshots_of_later_steps(self):
    self._write(global_step=3, anchor_step=6, prompt_ids=[12, 13])
    self._write(global_step=4, anchor_step=8, prompt_ids=[16])
    manager = self._manager()

    state = manager.restore(train_steps=7)

    self.assertEqual(state.global_step, 3)
    self.assertEqual(state.anchor_step, 6)
    self.assertEqual(
        [group[0].prompt_id for group in state.active_groups], [12, 13]
    )
    self.assertEqual(self._keys(manager), [6 * KEY_BASE, 6 * KEY_BASE + 1])
    manager.record_rollout_group(_group(14))
    manager.snapshot(wait=True)
    self.assertEqual(
        self._keys(manager),
        [6 * KEY_BASE, 6 * KEY_BASE + 1, 6 * KEY_BASE + 2],
    )
    self.assertLen(self._manager().restore(train_steps=7).active_groups, 3)

  def test_restore_raises_when_every_snapshot_is_newer(self):
    self._write(global_step=4, anchor_step=8, prompt_ids=[16])
    with self.assertRaisesRegex(ValueError, "after trainer step 7"):
      self._manager().restore(train_steps=7)

  def test_only_restored_groups_are_skipped(self):
    self._write(global_step=3, anchor_step=6, prompt_ids=[12])
    manager = self._manager()
    manager.restore(train_steps=6)
    manager.record_rollout_group(_group(13))
    self.assertTrue(manager.should_skip_group(12))
    self.assertFalse(manager.should_skip_group(13))
    # Restored ids stay skipped: the producer may reach them after the step.
    manager.start_step(global_step=4, anchor_step=8)
    self.assertTrue(manager.should_skip_group(12))
    self.assertEqual(manager.anchor_step, 8)

  def test_set_geometry_rejects_a_change_from_the_snapshot(self):
    self._write(global_step=3, anchor_step=6, prompt_ids=[12])
    manager = self._manager()
    manager.restore(train_steps=6)
    with self.assertRaisesRegex(ValueError, "geometry"):
      manager.set_geometry(dataclasses.replace(GEOMETRY, mini_batch_size=4))
    manager.set_geometry(GEOMETRY)


if __name__ == "__main__":
  absltest.main()
