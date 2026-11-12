"""Unit tests for Phase 3B: Freehand Brush / Eraser strokes and Undo."""

from __future__ import annotations

import json
import os
import unittest

from PIL import Image

import i18n_helper
import mask_ops
from ui import callbacks, tab


class TestPhase3BBrush(unittest.TestCase):
    def setUp(self):
        os.environ["SAM3_MASK_STUDIO_LANG"] = "ja_JP"
        i18n_helper.clear_cache()

    def tearDown(self):
        os.environ.pop("SAM3_MASK_STUDIO_LANG", None)
        i18n_helper.clear_cache()

    def test_brush_add_erase_and_undo(self):
        img = Image.new("RGB", (100, 100), (100, 100, 100))
        _prev, st, _chips, _msg = callbacks.on_image_Input_change(img, None)

        # Brush Add across (10, 25) -> (40, 25) on 50x50 display => (20, 50) -> (80, 50) on 100x100
        payload_add = json.dumps({
            "points": [[10, 25], [40, 25]],
            "radius": 5,
            "erase": False,
            "width": 50,
            "height": 50,
        })
        _p, st, _chips, _g, msg_add = callbacks.on_brush_stroke(payload_add, "Brush: Add", 5, st)
        self.assertIn("ストローク", msg_add)
        self.assertEqual(int(st.final_mask[50, 50]), 255)
        self.assertEqual(int(st.final_mask[0, 0]), 0)

        # Brush Erase at (25, 25) on 50x50 display => (50, 50) on 100x100
        payload_erase = json.dumps({
            "points": [[25, 25]],
            "radius": 4,
            "erase": True,
            "width": 50,
            "height": 50,
        })
        _p, st, _chips, _g, msg_erase = callbacks.on_brush_stroke(payload_erase, "Brush: Erase", 4, st)
        self.assertIn("ストローク", msg_erase)
        self.assertEqual(int(st.final_mask[50, 50]), 0)
        self.assertEqual(int(st.final_mask[50, 20]), 255)

        # Undo restores erased center
        _p, st, _m = callbacks.on_undo(st)
        self.assertEqual(int(st.final_mask[50, 50]), 255)

    def test_ui_tab_builds_with_phase3b_controls(self):
        blocks = tab.build_tab()
        self.assertIsNotNone(blocks)


if __name__ == "__main__":
    unittest.main()
