"""Comprehensive tests for M2 (Text Mask & Presets), M3 (Click Operations),
M4 (Grow/Shrink, Clear, Undo), M5 (Transfer & PNG Export), and M6 (BYOT Downloader & Localization).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

import i18n_helper
import mask_ops
import sam3_backend
import sam3_downloader
import transfer
from ui import callbacks, presets


class TestM2ToM6(unittest.TestCase):
    def setUp(self):
        os.environ["SAM3_MASK_STUDIO_LANG"] = "ja_JP"
        i18n_helper.clear_cache()

    def tearDown(self):
        os.environ.pop("SAM3_MASK_STUDIO_LANG", None)
        i18n_helper.clear_cache()

    # -----------------------------------------------------------------------
    # M2: Presets & Text Mask Set Operations
    # -----------------------------------------------------------------------
    def test_m2_presets_crud_and_localization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pfile = Path(tmpdir) / "presets.json"
            loaded = presets.load_presets(pfile)
            self.assertEqual(len(loaded), 4)
            choices = dict(presets.get_dropdown_choices(pfile))
            self.assertEqual(choices.get("人物全体"), "full_body")
            self.assertEqual(choices.get("髪"), "hair")

            # Custom preset falls back to its ID/name as display label
            presets.upsert_preset("海へと延びる直線的なコンクリート構造物", ["concrete pier"], ["sea"], path=pfile)
            custom = presets.find_preset("海へと延びる直線的なコンクリート構造物", path=pfile)
            self.assertIsNotNone(custom)
            self.assertEqual(custom["positive"], ["concrete pier"])
            self.assertEqual(custom["negative"], ["sea"])

            presets.delete_preset("海へと延びる直線的なコンクリート構造物", path=pfile)
            self.assertIsNone(presets.find_preset("海へと延びる直線的なコンクリート構造物", path=pfile))

    def test_m2_text_prompts_and_chip_removal(self):
        img = Image.new("RGB", (40, 40), (255, 255, 255))
        _prev, st, _chips_upd, _msg = callbacks.on_image_Input_change(img, None)

        def fake_predict_text(image, prompt, **kwargs):
            m = mask_ops.empty_mask((40, 40))
            if prompt == "hair":
                m[5:35, 5:35] = 255
            elif prompt == "face":
                m[15:25, 15:25] = 255
            elif prompt == "skin":
                m[25:30, 15:25] = 255
            return m

        with mock.patch("sam3_backend.predict_text_mask", side_effect=fake_predict_text):
            # Apply 'hair' preset -> positive: ['hair'], negative: ['face', 'skin']
            preview, st, chips_upd, cleared_in, status = callbacks.on_add_positive(
                "", "hair", st
            )
            self.assertIsNotNone(preview)
            self.assertEqual(cleared_in, "")
            self.assertEqual(st.positive_prompts, ["hair"])
            self.assertEqual(st.negative_prompts, ["face", "skin"])
            # Positive region outside negative should be 255; inside 'face' should be 0
            self.assertEqual(int(st.final_mask[10, 10]), 255)
            self.assertEqual(int(st.final_mask[20, 20]), 0)
            self.assertEqual(int(st.final_mask[27, 20]), 0)

            # Uncheck '- face' chip -> face exclusion is removed, so (20, 20) becomes 255!
            preview2, st, chips_upd2, status2 = callbacks.on_active_prompts_change(
                ["+ hair", "- skin"], st
            )
            self.assertEqual(st.negative_prompts, ["skin"])
            self.assertEqual(int(st.final_mask[20, 20]), 255)
            self.assertEqual(int(st.final_mask[27, 20]), 0)

    # -----------------------------------------------------------------------
    # M3: Click Add / Exclude & Coordinate Scaling
    # -----------------------------------------------------------------------
    def test_m3_click_add_and_exclude(self):
        img = Image.new("RGB", (100, 100), (200, 200, 200))
        _prev, st, _chips_upd, _msg = callbacks.on_image_Input_change(img, None)

        def fake_predict_point(image, point_coords, point_labels=None, **kwargs):
            x, y = point_coords[0]
            m = mask_ops.empty_mask((100, 100))
            if x < 50:
                m[10:60, 10:60] = 255
            else:
                m[30:80, 30:80] = 255
            return m

        with mock.patch("sam3_backend.predict_point_mask", side_effect=fake_predict_point):
            # Click Add at (10, 10) on a 50x50 display -> scales to (20, 20) on 100x100 image
            _p1, st, msg1 = callbacks.on_preview_click(
                "Click Mode: Add",
                st,
                coords=(10, 10),
                display_size=(50, 50),
            )
            self.assertEqual(st.click_points_pos, [(20, 20)])
            self.assertEqual(int(st.final_mask[20, 20]), 255)
            self.assertEqual(int(st.final_mask[40, 40]), 255)

            # Click Exclude at (35, 35) on 50x50 display -> scales to (70, 70)
            _p2, st, msg2 = callbacks.on_preview_click(
                "Click Mode: Exclude",
                st,
                coords=(35, 35),
                display_size=(50, 50),
            )
            self.assertEqual(st.click_points_neg, [(70, 70)])
            # Overlap region [30:60, 30:60] should now be subtracted (0)
            self.assertEqual(int(st.final_mask[20, 20]), 255)
            self.assertEqual(int(st.final_mask[40, 40]), 0)

    # -----------------------------------------------------------------------
    # M4: Grow / Shrink, Undo, Clear
    # -----------------------------------------------------------------------
    def test_m4_grow_shrink_undo_and_clear(self):
        img = Image.new("RGB", (50, 50), (100, 100, 100))
        _prev, st, _chips_upd, _msg = callbacks.on_image_Input_change(img, None)

        base_mask = mask_ops.empty_mask((50, 50))
        base_mask[20:30, 20:30] = 255
        st.positive_prompts = ["box"]
        st.pos_prompt_masks["box"] = base_mask
        callbacks.recompute_and_render(st, record_history=False)
        initial_area = int((st.final_mask > 0).sum())

        # Grow +3px
        _p, st, _m = callbacks.on_grow_change(3, st)
        grown_area = int((st.final_mask > 0).sum())
        self.assertGreater(grown_area, initial_area)

        # Undo -> reverts to initial_area
        _p, st, _m = callbacks.on_undo(st)
        self.assertEqual(int((st.final_mask > 0).sum()), initial_area)

        # Invert mask -> 50*50 - initial_area (2500 - 100 = 2400)
        _p, st, _chips, _grow, _m = callbacks.on_invert_mask(st)
        self.assertEqual(int((st.final_mask > 0).sum()), 2500 - initial_area)
        self.assertEqual(int(st.final_mask[0, 0]), 255)
        self.assertEqual(int(st.final_mask[25, 25]), 0)

        # Undo after Invert -> restores initial_area and prompt chips!
        _p, st, chips_upd, _grow, _thr, _m = callbacks.on_undo_with_ui(st)
        self.assertEqual(int((st.final_mask > 0).sum()), initial_area)
        self.assertEqual(chips_upd.get("value"), ["+ box"])

    def test_load_from_img2img_and_score_threshold(self):
        import base64
        import io

        src = Image.new("RGB", (32, 24), (50, 100, 150))
        buf = io.BytesIO()
        src.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        in_img, prev_img, st, _chips, msg = callbacks.on_load_from_img2img(data_url, None)
        self.assertEqual(in_img.size, (32, 24))
        self.assertEqual(prev_img.size, (32, 24))
        self.assertIn("32x24", msg)

        st.positive_prompts = ["person"]
        st.pos_prompt_masks["person"] = mask_ops.empty_mask((24, 32))

        with mock.patch("sam3_backend.predict_text_mask") as mock_pred:
            m = mask_ops.empty_mask((24, 32))
            m[2:10, 2:10] = 255
            mock_pred.return_value = m
            _prev2, st, thr_msg = callbacks.on_score_threshold_change(0.55, st)
            self.assertAlmostEqual(st.score_threshold, 0.55)
            mock_pred.assert_called_once()
            self.assertEqual(int(st.final_mask[5, 5]), 255)

        # Clear mask -> becomes 0, then Undo restores it
        _p, st, _chips, grow_val, _m = callbacks.on_clear_mask(st)
        self.assertEqual(grow_val, 0)
        self.assertEqual(int((st.final_mask > 0).sum()), 0)

        _p, st, _m = callbacks.on_undo(st)
        self.assertEqual(int((st.final_mask > 0).sum()), initial_area)

        # Invert mask -> 50*50 - initial_area (2500 - 100 = 2400)
        _p, st, _chips, _grow, _m = callbacks.on_invert_mask(st)
        self.assertEqual(int((st.final_mask > 0).sum()), 2500 - initial_area)
        self.assertEqual(int(st.final_mask[0, 0]), 255)
        self.assertEqual(int(st.final_mask[25, 25]), 0)

        # Undo after Invert -> restores initial_area and prompt chips!
        _p, st, chips_upd, _grow, _thr, _m = callbacks.on_undo_with_ui(st)
        self.assertEqual(int((st.final_mask > 0).sum()), initial_area)
        self.assertEqual(chips_upd.get("value"), ["+ box"])

    def test_load_from_img2img_and_score_threshold(self):
        import base64
        import io

        src = Image.new("RGB", (32, 24), (50, 100, 150))
        buf = io.BytesIO()
        src.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        in_img, prev_img, st, _chips, msg = callbacks.on_load_from_img2img(data_url, None)
        self.assertEqual(in_img.size, (32, 24))
        self.assertEqual(prev_img.size, (32, 24))
        self.assertIn("32x24", msg)

        st.positive_prompts = ["person"]
        st.pos_prompt_masks["person"] = mask_ops.empty_mask((24, 32))

        with mock.patch("sam3_backend.predict_text_mask") as mock_pred:
            m = mask_ops.empty_mask((24, 32))
            m[2:10, 2:10] = 255
            mock_pred.return_value = m
            _prev2, st, thr_msg = callbacks.on_score_threshold_change(0.55, st)
            self.assertAlmostEqual(st.score_threshold, 0.55)
            mock_pred.assert_called_once()
            self.assertEqual(int(st.final_mask[5, 5]), 255)

    # -----------------------------------------------------------------------
    # M5: Inpaint Upload Transfer & Masked RGBA PNG Export
    # -----------------------------------------------------------------------
    def test_m5_transfer_and_masked_png_export(self):
        img = Image.new("RGB", (40, 30), (10, 120, 240))
        _prev, st, _chips, _msg = callbacks.on_image_Input_change(img, None)
        m = mask_ops.empty_mask((30, 40))
        m[5:15, 10:25] = 255
        st.final_mask = m

        # 1. Inpaint transfer with component updates
        res = callbacks.on_send_to_inpaint(st, has_inpaint_components=True)
        self.assertEqual(len(res), 4)
        msg, payload_json, base_upd, mask_upd = res
        payload = json.loads(payload_json)
        self.assertIn("image", payload)
        self.assertIn("mask", payload)
        self.assertEqual(base_upd["value"].size, (40, 30))
        self.assertEqual(mask_upd["value"].mode, "RGBA")

        # 2. Masked PNG Export (Full + Transparent vs Crop + White Background)
        with tempfile.TemporaryDirectory() as tmpdir:
            msg_full, payload_full = callbacks.on_download_masked_png(
                "Full Image", "Transparent", st, save_dir=tmpdir
            )
            data_full = json.loads(payload_full)
            full_path = Path(tmpdir) / data_full["filename"]
            self.assertTrue(full_path.is_file())
            with Image.open(full_path) as exported_full:
                self.assertEqual(exported_full.mode, "RGBA")
                self.assertEqual(exported_full.size, (40, 30))
                # Outside mask alpha == 0, inside mask alpha == 255
                self.assertEqual(exported_full.getpixel((0, 0))[3], 0)
                self.assertEqual(exported_full.getpixel((15, 10))[3], 255)

            msg_crop, payload_crop = callbacks.on_download_masked_png(
                "Crop to Mask", "White Background", st, save_dir=tmpdir
            )
            data_crop = json.loads(payload_crop)
            crop_path = Path(tmpdir) / data_crop["filename"]
            with Image.open(crop_path) as exported_crop:
                self.assertEqual(exported_crop.size, (15, 10))
                self.assertEqual(exported_crop.getpixel((0, 0)), (10, 120, 240, 255))

    # -----------------------------------------------------------------------
    # M6: BYOT Access Check & Error Handling
    # -----------------------------------------------------------------------
    def test_m6_byot_downloader_error_handling(self):
        with mock.patch("sam3_downloader.HfApi") as mock_cls:
            mock_cls.return_value.model_info.return_value = object()
            code, msg = sam3_downloader.check_access(token="hf_test")
            self.assertEqual(code, "access_granted")
            self.assertIn("アクセス確認OK", msg)

        with mock.patch("sam3_downloader.HfApi") as mock_cls:
            resp = mock.MagicMock()
            resp.status_code = 403
            mock_cls.return_value.model_info.side_effect = GatedRepoError("gated", response=resp)
            code, msg = sam3_downloader.check_access(token="hf_test")
            self.assertEqual(code, "gated_not_accepted")
            self.assertIn("ライセンス未承認", msg)

        with mock.patch("sam3_downloader.HfApi") as mock_cls:
            resp = mock.MagicMock()
            resp.status_code = 404
            mock_cls.return_value.model_info.side_effect = RepositoryNotFoundError("not found", response=resp)
            code, msg = sam3_downloader.check_access(token="hf_test")
            self.assertEqual(code, "not_found")
            self.assertIn("トークン無効", msg)

    # -----------------------------------------------------------------------
    # Real SAM3 GPU Backend Integration Test
    # -----------------------------------------------------------------------
    def test_real_sam3_backend_text_and_click(self):
        img = Image.new("RGB", (256, 256), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([64, 64, 192, 192], fill=(220, 30, 30))

        tmask = sam3_backend.predict_text_mask(img, "red square")
        self.assertEqual(tmask.shape, (256, 256))
        self.assertEqual(tmask.dtype, np.uint8)
        self.assertEqual(int(tmask[128, 128]), 255)
        self.assertEqual(int(tmask[10, 10]), 0)

        pmask = sam3_backend.predict_point_mask(img, [(128, 128)], [1])
        self.assertEqual(pmask.shape, (256, 256))
        self.assertEqual(pmask.dtype, np.uint8)
        self.assertEqual(int(pmask[128, 128]), 255)
        self.assertEqual(int(pmask[10, 10]), 0)

        ok, _msg = sam3_backend.unload_model()
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
