"""Unit tests for lightweight bounding-box helpers."""

from __future__ import annotations

import unittest

from tests.test_support import import_project_module


class BboxUtilsTests(unittest.TestCase):
    """Exercise pure helpers that do not require TensorFlow ops."""

    def setUp(self) -> None:
        self.module, _ = import_project_module(
            "utils.bbox_utils",
            clear_modules=("utils.bbox_utils",),
        )

    def test_get_scale_for_nth_feature_map_matches_ssd_formula(self) -> None:
        scale = self.module.get_scale_for_nth_feature_map(3)

        self.assertAlmostEqual(scale, 0.48)


if __name__ == "__main__":
    unittest.main()
