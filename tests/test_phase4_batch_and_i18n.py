"""Unit tests for Phase 4: Batch Processing and i18n completeness verification."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import i18n_helper
import mask_ops
from ui import callbacks, tab

_ROOT = Path(__file__).resolve().parents[1]


class TestPhase4BatchAndI18n(unittest.TestCase):
    def setUp(self):
        os.environ["SAM3_MASK_STUDIO_LANG"] = "ja_JP"
        i18n_helper.clear_cache()

    def tearDown(self):
        os.environ.pop("SAM3_MASK_STUDIO_LANG", None)
        i18n_helper.clear_cache()

    def test_batch_process_validation_and_execution(self):
        self.assertIn("入力フォルダ", callbacks.on_batch_process("", "", "person", "", None, 0, 0, 0.35, "Full Image", "Transparent"))
        self.assertIn("見つかりません", callbacks.on_batch_process("/nonexistent_sam3_dir_xyz", "", "person", "", None, 0, 0, 0.35, "Full Image", "Transparent"))

        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            # Empty directory
            self.assertIn("画像が見つかりません", callbacks.on_batch_process(in_dir, out_dir, "person", "", None, 0, 0, 0.35, "Full Image", "Transparent"))

            # Create 2 sample images
            Image.new("RGB", (60, 60), (10, 20, 30)).save(Path(in_dir) / "img1.png")
            Image.new("RGB", (60, 60), (40, 50, 60)).save(Path(in_dir) / "img2.jpg")

            # Missing prompt
            self.assertIn("プロンプト", callbacks.on_batch_process(in_dir, out_dir, "", "", None, 0, 0, 0.35, "Full Image", "Transparent"))

            def fake_predict_text(image, prompt, **kwargs):
                m = mask_ops.empty_mask((image.height, image.width))
                if prompt == "person":
                    m[10:50, 10:50] = 255
                return m

            with mock.patch("sam3_backend.predict_text_mask", side_effect=fake_predict_text):
                msg = callbacks.on_batch_process(
                    in_dir,
                    out_dir,
                    "person",
                    "",
                    None,
                    0,
                    2,
                    0.35,
                    "Crop to Mask",
                    "Mask Only (B&W)",
                )
            self.assertIn("2 件保存", msg)
            out1 = Path(out_dir) / "img1.png"
            out2 = Path(out_dir) / "img2.png"
            self.assertTrue(out1.is_file())
            self.assertTrue(out2.is_file())
            with Image.open(out1) as res:
                self.assertEqual(res.mode, "RGB")
                self.assertGreater(res.width, 0)

    def test_i18n_ja_jp_completeness(self):
        ja_path = _ROOT / "localizations" / "ja_JP.json"
        ja_data = json.loads(ja_path.read_text(encoding="utf-8"))
        self.assertIsInstance(ja_data, dict)

        pattern = re.compile(r'\bt\(\s*["\']([^"\']+)["\']')
        missing: list[str] = []
        for rel in ("ui/callbacks.py", "transfer.py", "sam3_downloader.py", "ui/presets.py"):
            p = _ROOT / rel
            if not p.is_file():
                continue
            txt = p.read_text(encoding="utf-8")
            for key in pattern.findall(txt):
                if key not in ja_data:
                    missing.append(f"{rel}: {key}")

        self.assertEqual(missing, [], f"Missing ja_JP.json keys: {missing}")

    def test_ui_tab_builds_with_phase4_controls(self):
        blocks = tab.build_tab()
        self.assertIsNotNone(blocks)


if __name__ == "__main__":
    unittest.main()
