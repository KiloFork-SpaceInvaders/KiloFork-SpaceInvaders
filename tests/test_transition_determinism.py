"""Headless producer contract for the Player Two data-only interchange."""
import json
import tempfile
import unittest
from pathlib import Path

from space_invaders.core import Action
from space_invaders.training import run_episode


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
            self.assertEqual(a["feature_count"], 134)
            for item in entries:
                self.assertEqual(item["schema"], "kilocore.shmup-transition.v1")
                self.assertEqual(item["feature_count"], 134)
                self.assertEqual(len(item["observation"]), 134)
                self.assertEqual(len(item["next_observation"]), 134)
                self.assertEqual(item["action"], int(Action.FIRE))
                self.assertIn("reward", item)
                self.assertIsInstance(item["terminated"], bool)

    def test_action_change_changes_serialized_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            firing, idle = Path(tmp, "firing.jsonl"), Path(tmp, "idle.jsonl")
            run_episode(ConstantPolicy(Action.FIRE), seed=144, max_ticks=32, record_path=firing)
            run_episode(ConstantPolicy(Action.NONE), seed=144, max_ticks=32, record_path=idle)
            self.assertNotEqual(firing.read_bytes(), idle.read_bytes())
            self.assertEqual(_recorded(firing)[0]["action"], int(Action.FIRE))
            self.assertEqual(_recorded(idle)[0]["action"], int(Action.NONE))


if __name__ == "__main__":
    unittest.main()
