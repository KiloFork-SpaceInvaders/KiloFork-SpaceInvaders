"""Headless producer contract for the Player Two data-only interchange."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from space_invaders.core import Action
from space_invaders.features import ACTION_SCHEMA, VECTOR_OBSERVATION_SCHEMA, VECTOR_FEATURE_COUNT
from space_invaders.training import Transition, TransitionRecorder, run_episode


class ConstantPolicy:
    def __init__(self, action: Action):
        self.action = action

    def act(self, _observation):
        return self.action


def _recorded(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class EpisodeRecordContract(unittest.TestCase):
    def test_same_seed_same_actions_identical_transition_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp, "one.jsonl"), Path(tmp, "two.jsonl")
            a = run_episode(ConstantPolicy(Action.FIRE), seed=144, max_ticks=48, record_path=one)
            b = run_episode(ConstantPolicy(Action.FIRE), seed=144, max_ticks=48, record_path=two)
            self.assertEqual(a, b)
            self.assertEqual(one.read_bytes(), two.read_bytes())
            entries = _recorded(one)
            self.assertTrue(entries)
            self.assertEqual(len(entries), a["ticks"])
            self.assertEqual(a["schema"], "kilocore.shmup-episode.v1")
            self.assertEqual(a["feature_count"], VECTOR_FEATURE_COUNT)
            self.assertEqual(a["observation_schema"], VECTOR_OBSERVATION_SCHEMA)
            self.assertEqual(a["action_schema"], ACTION_SCHEMA)
            self.assertAlmostEqual(a["total_reward"], 48 * 0.002, places=9)
            self.assertFalse(a["terminated"], "48-tick fixture has not ended the game")
            for item in entries:
                self.assertEqual(item["schema"], "kilocore.shmup-transition.v1")
                self.assertEqual(item["feature_count"], VECTOR_FEATURE_COUNT)
                self.assertEqual(item["observation_schema"], VECTOR_OBSERVATION_SCHEMA)
                self.assertEqual(item["action_schema"], ACTION_SCHEMA)
                for vector_key in ("observation", "next_observation"):
                    vector = item[vector_key]
                    self.assertEqual(len(vector), VECTOR_FEATURE_COUNT)
                    self.assertTrue(all(type(value) in (int, float) and math.isfinite(value)
                                        for value in vector), f"{vector_key} must be numeric and finite")
                self.assertEqual(item["action"], int(Action.FIRE))
                self.assertEqual(item["reward"], 0.002)
                self.assertIs(item["terminated"], False)

    def test_action_change_changes_observed_state_not_just_labels(self):
        # Opposing human-control inputs move the player apart deterministically.
        # Comparing raw JSON would pass even with a broken engine that ignored
        # actions, because each row includes the chosen action label.
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp, "left.jsonl"), Path(tmp, "right.jsonl")
            run_episode(ConstantPolicy(Action.LEFT), seed=144, max_ticks=32, record_path=left)
            run_episode(ConstantPolicy(Action.RIGHT), seed=144, max_ticks=32, record_path=right)
            lrows, rrows = _recorded(left), _recorded(right)
            self.assertEqual(len(lrows), len(rrows))
            self.assertTrue(lrows)
            self.assertEqual(lrows[0]["action"], int(Action.LEFT))
            self.assertEqual(rrows[0]["action"], int(Action.RIGHT))
            l_state = [row["next_observation"] for row in lrows]
            r_state = [row["next_observation"] for row in rrows]
            self.assertNotEqual(l_state, r_state, "different inputs must change observed state")
            self.assertLess(l_state[0][0], r_state[0][0], "LEFT vs RIGHT must move x oppositely")

    def test_recorder_preserves_terminal_and_negative_reward(self):
        # A controlled recorder fixture tests the serialization boundary.
        # It is NOT evidence of a terminal event from the actual game engine.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "terminal.jsonl")
            recorder = TransitionRecorder(path)
            recorder.write(Transition(
                observation=[0.0] * VECTOR_FEATURE_COUNT,
                action=int(Action.NONE),
                reward=-20.0,
                next_observation=[0.0] * VECTOR_FEATURE_COUNT,
                terminated=True,
                info={"tick": 1, "lives": 0},
            ))
            rows = _recorded(path)
            self.assertEqual(len(rows), 1)
            self.assertIs(rows[0]["terminated"], True)
            self.assertEqual(rows[0]["reward"], -20.0)
            self.assertEqual(rows[0]["observation_schema"], VECTOR_OBSERVATION_SCHEMA)
            self.assertEqual(rows[0]["action_schema"], ACTION_SCHEMA)


if __name__ == "__main__":
    unittest.main()
