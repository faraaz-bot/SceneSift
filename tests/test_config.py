"""Configuration loading and validation."""
import unittest
from dataclasses import fields
from pathlib import Path

from scenesift.config import FEATURES, METHOD_FEATURES, Config

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class ConfigTests(unittest.TestCase):
    def test_default_file_matches_code_defaults(self):
        self.assertEqual(Config.from_file(DEFAULT_FILE), Config())

    def test_default_file_lists_every_setting(self):
        import yaml

        listed = set(yaml.safe_load(DEFAULT_FILE.read_text()))
        self.assertEqual(listed, {f.name for f in fields(Config)})

    def test_partial_weights_keep_the_other_defaults(self):
        cfg = Config.from_dict({"weights": {"motion": 0.5}})
        self.assertEqual(cfg.weights["motion"], 0.5)
        self.assertEqual(cfg.weights["visual"], Config().weights["visual"])

    def test_unknown_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"wieghts": {}})
        with self.assertRaises(ValueError):
            Config.from_dict({"weights": {"smell": 1.0}})

    def test_methods_only_use_known_features(self):
        for used in METHOD_FEATURES.values():
            self.assertLessEqual(set(used), set(FEATURES))


if __name__ == "__main__":
    unittest.main()
