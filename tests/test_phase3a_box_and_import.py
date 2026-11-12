"""Unit tests for Phase 3A: Base Mask Import and Shift+Drag Bounding-Box Selection."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import i18n_helper
import mask_ops
from ui import callbacks, tab


class TestPhase3ABoxAndImport(unittest.TestCase):
    def setUp(self):
        os.environ["SAM3_MASK_STUDIO_LANG"] = "ja_JP"
        i18n_helper.clear_cache()

    def tearDown(self):
        os.environ.pop("SAM3_MASK_STUDIO_LANG", None)
        i18n_helper.clear_cache()

    def test_import_base_mask_resizes_and_supports_undo(self):
        img = Image.new("RGB", (80, 60), (50, 50, 50))
        _prev, st, _chips, _msg = callbacks.on_image_Input_change(img, None)

        # Create a 40x30 mask PNG file and import it
        m_img = Image.new("L", (40, 30), 0)
        arr = np.zeros((30, 40), dtype=np.uint8)
        arr[10:20, 10:30] = 255
        m_img = Image.fromarray(arr, mode="L")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp_path = tf.name
        try:
            m_img.save(tmp_path)
            _p, st, msg = callbacks.on_import_base_mask(tmp_path, st)
            self.assertIn("80x60", msg)
            self.assertEqual(st.final_mask.shape, (60, 80))
            self.assertEqual(int(st.final_mask[30, 40]), 255)
            self.assertEqual(int(st.final_mask[0, 0]), 0)

            # Undo restores empty mask
            _p, st, _m = callbacks.on_undo(st)
            self.assertFalse(st.final_mask.any())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_box_select_add_and_exclude(self):
        img = Image.new("RGB", (100, 100), (200, 200, 200))
        _prev, st, _chips, _msg = callbacks.on_image_Input_change(img, None)

        def fake_predict_box(image, box_xyxy, **kwargs):
            x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy]
            m = mask_ops.empty_mask((100, 100))
            m[y0:y1, x0:x1] = 255
            return m

        with mock.patch("sam3_backend.predict_box_mask", side_effect=fake_predict_box):
            payload_add = json.dumps({"x0": 5, "y0": 5, "x1": 25, "y1": 25, "width": 50, "height": 50})
            _p, st, msg_add = callbacks.on_box_select(payload_add, "Click Mode: Add", st)
            self.assertIn("矩形選択", msg_add)
            # Scaled x2 -> [10:50, 10:50]
            self.assertEqual(int(st.final_mask[20, 20]), 255)
            self.assertEqual(int(st.final_mask[30, 30]), 255)

            payload_excl = json.dumps({"x0": 10, "y0": 10, "x1": 15, "y1": 15, "width": 50, "height": 50})
            _p, st, msg_excl = callbacks.on_box_select(payload_excl, "Click Mode: Exclude", st)
            self.assertIn("矩形選択", msg_excl)
            # Scaled x2 -> [20:30, 20:30] excluded
            self.assertEqual(int(st.final_mask[25, 25]), 0)
            self.assertEqual(int(st.final_mask[15, 15]), 255)

            # Undo restores excluded region
            _p, st, _m = callbacks.on_undo(st)
            self.assertEqual(int(st.final_mask[25, 25]), 255)

    def test_ui_tab_builds_with_phase3a_controls(self):
        blocks = tab.build_tab()
        self.assertIsNotNone(blocks)


if __name__ == "__main__":
    unittest.main()
