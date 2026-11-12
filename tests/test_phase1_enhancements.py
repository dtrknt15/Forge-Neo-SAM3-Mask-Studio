"""Unit tests for Phase 1 enhancements: Feather, Fill Holes, Remove Small Islands, and Mask-Only PNG export."""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

import i18n_helper
import mask_ops
import transfer
from ui import callbacks, tab


class TestPhase1Enhancements(unittest.TestCase):
    def setUp(self):
        os.environ["SAM3_MASK_STUDIO_LANG"] = "ja_JP"
        i18n_helper.clear_cache()

    def tearDown(self):
        os.environ.pop("SAM3_MASK_STUDIO_LANG", None)
        i18n_helper.clear_cache()

    def test_feather_fill_holes_and_remove_islands(self):
        # Square with an interior hole and a tiny 4px island
        m = mask_ops.empty_mask((64, 64))
        m[0:32, 0:32] = 255  # touches top-left border
        m[10:18, 10:18] = 0  # enclosed hole
        m[50:52, 50:52] = 255  # 4px island

        filled = mask_ops.fill_holes(m)
        self.assertEqual(int(filled[14, 14]), 255)
        # Exterior should remain 0
        self.assertEqual(int(filled[40, 40]), 0)

        cleaned = mask_ops.remove_small_islands(filled, min_area=16)
        self.assertEqual(int(cleaned[14, 14]), 255)
        self.assertEqual(int(cleaned[50, 50]), 0)

        feathered = mask_ops.apply_feather(cleaned, feather_px=4)
        self.assertEqual(feathered.dtype, np.uint8)
        # Boundary should now have intermediate soft values (0 < val < 255)
        edge_val = int(feathered[32, 16])
        self.assertGreater(edge_val, 0)
        self.assertLess(edge_val, 255)

    def test_callbacks_feather_holes_islands_and_undo(self):
        img = Image.new("RGB", (64, 64), (120, 140, 160))
        _prev, st, _chips, _msg = callbacks.on_image_Input_change(img, None)

        m = mask_ops.empty_mask((64, 64))
        m[10:50, 10:50] = 255
        m[20:30, 20:30] = 0  # hole
        m[2:4, 2:4] = 255    # 4px island
        st.base_mask = m
        callbacks.recompute_and_render(st)

        # Fill holes
        _p, st, _c, _g, msg_holes = callbacks.on_fill_holes(st)
        self.assertIn("穴埋め", msg_holes)
        self.assertEqual(int(st.final_mask[25, 25]), 255)

        # Remove small islands
        _p, st, _c, _g, msg_isl = callbacks.on_remove_small_islands(16, st)
        self.assertIn("小ノイズ", msg_isl)
        self.assertEqual(int(st.final_mask[2, 2]), 0)

        # Feather
        _p, st, msg_feather = callbacks.on_feather_change(6, st)
        self.assertIn("6", msg_feather)
        self.assertEqual(st.feather_px, 6)

        # Undo restores feather_px=0
        _p, st, _msg = callbacks.on_undo(st)
        self.assertEqual(st.feather_px, 0)

        # Undo restores 4px island
        _p, st, _msg = callbacks.on_undo(st)
        self.assertEqual(int(st.final_mask[2, 2]), 255)

        # Undo restores interior hole
        _p, st, _msg = callbacks.on_undo(st)
        self.assertEqual(int(st.final_mask[25, 25]), 0)

    def test_inpaint_preserves_feather_and_mask_only_export(self):
        img = Image.new("RGB", (64, 64), (200, 100, 50))
        _prev, st, _chips, _msg = callbacks.on_image_Input_change(img, None)
        m = mask_ops.empty_mask((64, 64))
        m[16:48, 16:48] = 255
        st.base_mask = m
        callbacks.recompute_and_render(st)
        _p, st, _m = callbacks.on_feather_change(6, st)

        # Send to Inpaint should preserve soft edge values
        msg, payload, init_upd, mask_upd = callbacks.on_send_to_inpaint(
            st, has_inpaint_components=True
        )
        self.assertIn("Inpaint", msg)
        mask_arr = np.asarray(mask_upd["value"])
        edge_r = int(mask_arr[16, 32, 0])
        self.assertGreater(edge_r, 0)
        self.assertLess(edge_r, 255)

        # Mask Only (B&W) export
        with tempfile.TemporaryDirectory() as tmpdir:
            msg_bw, payload_bw = callbacks.on_download_masked_png(
                "Crop to Mask",
                "マスクのみ (白黒)",
                st,
                save_dir=tmpdir,
            )
            self.assertIn("書き出しました", msg_bw)
            data = json.loads(payload_bw)
            self.assertTrue(data["filename"].startswith("sam3_mask_bw_"))
            raw = base64.b64decode(data["image"])
            with Image.open(io.BytesIO(raw)) as exported:
                self.assertEqual(exported.mode, "RGB")
                arr = np.asarray(exported)
                # Center should be pure white (255, 255, 255)
                cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
                self.assertEqual(tuple(int(v) for v in arr[cy, cx]), (255, 255, 255))

    def test_ui_tab_builds_with_phase1_controls(self):
        blocks = tab.build_tab()
        self.assertIsNotNone(blocks)


if __name__ == "__main__":
    unittest.main()
