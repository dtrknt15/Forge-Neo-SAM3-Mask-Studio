"""Unit tests for v1 improvements: threshold empty fallback, batch options, auto-unload, gallery send."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import numpy as np
from PIL import Image

import i18n_helper
import mask_ops
import sam3_backend
import script
from ui import callbacks


class TestImprovementsV1(unittest.TestCase):
    def setUp(self):
        os.environ["SAM3_MASK_STUDIO_LANG"] = "ja_JP"
        i18n_helper.clear_cache()

    def tearDown(self):
        os.environ.pop("SAM3_MASK_STUDIO_LANG", None)
        i18n_helper.clear_cache()
    def test_extract_binary_masks_returns_empty_when_below_threshold(self):
        m1 = np.zeros((16, 16), dtype=np.float32)
        m1[2:8, 2:8] = 1.0
        state = {
            "masks": np.stack([m1], axis=0),
            "scores": np.array([0.15], dtype=np.float32),
        }
        res = sam3_backend._extract_binary_masks_from_state(
            state, target_shape=(16, 16), score_threshold=0.50
        )
        self.assertEqual(res.shape, (16, 16))
        self.assertFalse(res.any())

    def test_batch_process_skip_empty_and_postprocess_options(self):
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            Image.new("RGB", (32, 32), (10, 20, 30)).save(Path(in_dir) / "hit.png")
            Image.new("RGB", (32, 32), (40, 50, 60)).save(Path(in_dir) / "miss.png")

            call_idx = {"n": 0}
            progress_calls = []

            def fake_predict(image, prompt, **kwargs):
                call_idx["n"] += 1
                m = mask_ops.empty_mask((image.height, image.width))
                if call_idx["n"] == 1:
                    # 10x10 square with a 2x2 hole in center
                    m[5:15, 5:15] = 255
                    m[9:11, 9:11] = 0
                return m

            def fake_progress(val, desc=""):
                progress_calls.append((val, desc))

            with mock.patch("sam3_backend.predict_text_mask", side_effect=fake_predict):
                msg = callbacks.on_batch_process(
                    in_dir,
                    out_dir,
                    "person",
                    "",
                    None,
                    0,
                    0,
                    0.35,
                    "Full Image",
                    "Mask Only (B&W)",
                    fill_holes_opt=True,
                    remove_islands_opt=True,
                    min_island_area=4,
                    invert_opt=True,
                    skip_empty=True,
                    progress=fake_progress,
                )

            self.assertEqual(len(progress_calls), 2)
            self.assertTrue((Path(out_dir) / "hit.png").is_file())
            self.assertFalse((Path(out_dir) / "miss.png").exists())
            self.assertIn("1", msg)
            with Image.open(Path(out_dir) / "hit.png") as res:
                arr = np.array(res.convert("L"))
                # Hole was filled before invert -> center of square is 0 after invert
                self.assertEqual(int(arr[10, 10]), 0)
                # Outside square was 0 -> 255 after invert
                self.assertEqual(int(arr[0, 0]), 255)

    def test_send_to_inpaint_auto_unload(self):
        st = callbacks.MaskStudioState()
        img = Image.new("RGB", (24, 24), (100, 100, 100))
        st.reset_for_new_image(img)
        st.final_mask[4:12, 4:12] = 255

        with (
            mock.patch("sam3_backend.is_loaded", return_value=True),
            mock.patch("sam3_backend.unload_model", return_value=(True, "ok")) as unload_mock,
        ):
            msg, payload = callbacks.on_send_to_inpaint(
                st, has_inpaint_components=False, auto_unload=True
            )
            unload_mock.assert_called_once()
            self.assertTrue(bool(payload))
            self.assertIn("24x24", msg)

    def test_gallery_send_button_hook_and_js_export(self):
        with gr.Blocks():
            with gr.Row():
                dummy = gr.Button("Inpaint", elem_id="txt2img_send_to_inpaint")
                script.on_after_component(dummy, elem_id="txt2img_send_to_inpaint")
        js_path = Path(__file__).resolve().parents[1] / "javascript" / "sam3_mask_studio.js"
        js_text = js_path.read_text(encoding="utf-8")
        self.assertIn("sam3MaskStudioSendFromGallery", js_text)


if __name__ == "__main__":
    unittest.main()
