import glob
import os
import unittest

from twbacktest.config import load_config
from twbacktest.grid import expand

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
# 每個範例展開後的策略組數（手寫字面值）
EXPECTED = {"basic.toml": 3}


class ExamplesMatchConfigFormat(unittest.TestCase):
    def test_every_example_loads_and_expands(self):
        files = sorted(glob.glob(os.path.join(EXAMPLES, "*.toml")))
        self.assertTrue(files, "examples/ 下沒有任何 .toml")
        self.assertEqual({os.path.basename(f) for f in files}, set(EXPECTED))
        for f in files:
            with self.subTest(example=os.path.basename(f)):
                self.assertEqual(len(expand(load_config(f))), EXPECTED[os.path.basename(f)])


if __name__ == "__main__":
    unittest.main()
