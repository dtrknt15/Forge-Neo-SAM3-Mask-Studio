"""Unit tests for Phase 2 enhancements: Preview View modes and Overlay Opacity."""

from __future__ import annotations

import os
import unittest

import numpy as np
from PIL import Image

import i18n_helper
import mask_ops
from ui import callbacks, tab


class TestPhase2Preview(unittest.TestCase):
    def setUp(self):
        os.environ["SAM3_MASK_STUDIO_LANG"] = "ja_JP"
        i18n_helper.clear_cache()

    def tearDown(self):
        os.environ.pop("SAM3_MASK_STUDIO_LANG", None)
        i18n_helper.clear_cache()

    def test_preview_modes_and_opacity(self):
        img = Image.new("RGB", (40, 40), (100, 100, 100))
        _prev, st, _chips, _msg = callbacks.on_image_Input_change(img, None)

        m = mask_ops.empty_mask((40, 40))
        m[10:30, 10:30] = 255
        st.base_mask = m
        callbacks.recompute_and_render(st)

        # Original view: should match base image pixels
        orig_prev, st = callbacks.on_preview_mode_change("元画像", st)
        self.assertEqual(orig_prev.getpixel((20, 20)), (100, 100, 100))

        # Mask Only view: white inside mask, black outside
        mask_prev, st = callbacks.on_preview_mode_change("マスク単体", st)
        self.assertEqual(mask_prev.getpixel((20, 20)), (255, 255, 255))
        self.assertEqual(mask_prev.getpixel((0, 0)), (0, 0, 0))

        # Overlay view with custom opacity
        ov_low, st = callbacks.on_preview_mode_change("Overlay", st)
        ov_low, st = callbacks.on_overlay_opacity_change(0.20, st)
        ov_high, st = callbacks.on_overlay_opacity_change(0.80, st)
        # Higher opacity should shift masked pixel further toward DEFAULT_FINAL_COLOR (0, 200, 255)
        self.assertGreater(ov_high.getpixel((20, 20))[2], ov_low.getpixel((20, 20))[2])
        self.assertLess(ov_high.getpixel((20, 20))[0], ov_low.getpixel((20, 20))[0])

    def test_ui_tab_builds_with_phase2_controls(self):
        blocks = tab.build_tab()
        self.assertIsNotNone(blocks)


if __name__ == "__main__":
    unittest.main()
