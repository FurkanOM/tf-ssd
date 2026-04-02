"""Unit tests for training helper utilities."""

from __future__ import annotations

import copy
import unittest

from tests.test_support import import_project_module


class TrainUtilsTests(unittest.TestCase):
    """Cover hyper-parameter and batching helpers."""

    def setUp(self) -> None:
        self.module, _ = import_project_module(
            "utils.train_utils",
            clear_modules=("utils.train_utils", "utils.bbox_utils"),
        )
        original_ssd = copy.deepcopy(self.module.SSD)

        def restore_ssd() -> None:
            self.module.SSD.clear()
            self.module.SSD.update(copy.deepcopy(original_ssd))

        self.addCleanup(restore_ssd)

    def test_get_hyper_params_populates_defaults(self) -> None:
        params = self.module.get_hyper_params("vgg16")

        self.assertEqual(params["img_size"], 300)
        self.assertEqual(params["iou_threshold"], 0.5)
        self.assertEqual(params["neg_pos_ratio"], 3)
        self.assertEqual(params["loc_loss_alpha"], 1)
        self.assertEqual(params["variances"], [0.1, 0.1, 0.2, 0.2])

    def test_get_hyper_params_applies_truthy_overrides(self) -> None:
        params = self.module.get_hyper_params(
            "mobilenet_v2",
            img_size=320,
            iou_threshold=0.6,
        )

        self.assertEqual(params["img_size"], 320)
        self.assertEqual(params["iou_threshold"], 0.6)

    def test_get_hyper_params_returns_a_copy_of_backbone_defaults(self) -> None:
        params = self.module.get_hyper_params("vgg16")

        params["img_size"] = 320

        self.assertEqual(self.module.SSD["vgg16"]["img_size"], 300)

    def test_scheduler_steps_down_at_expected_epochs(self) -> None:
        self.assertEqual(self.module.scheduler(0), 1e-3)
        self.assertEqual(self.module.scheduler(110), 1e-4)
        self.assertEqual(self.module.scheduler(130), 1e-5)

    def test_get_step_size_rounds_up_for_partial_batches(self) -> None:
        self.assertEqual(self.module.get_step_size(10, 4), 3)
        self.assertEqual(self.module.get_step_size(8, 4), 2)


if __name__ == "__main__":
    unittest.main()
