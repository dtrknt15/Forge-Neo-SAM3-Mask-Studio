"""Unit tests for M1 Foundation (install, i18n_helper, mask_ops, ui/tab, script)."""

from __future__ import annotations

import os
import unittest
import numpy as np
from PIL import Image

import i18n_helper
import install
import mask_ops
import script
from ui import callbacks


class TestM1Foundation(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("SAM3_MASK_STUDIO_LANG", None)
        i18n_helper.clear_cache()

    def test_install_checks(self):
        self.assertIsNone(install.check_python())
        self.assertIsNone(install.check_torch())
        self.assertTrue(install.check_model_exists())

    def test_i18n_helper_en_and_ja(self):
        i18n_helper.clear_cache()
        os.environ["SAM3_MASK_STUDIO_LANG"] = "en_US"
        self.assertEqual(i18n_helper.t("SAM3 Mask"), "SAM3 Mask")
        self.assertEqual(i18n_helper.t("status.image_loaded", width=512, height=768), "status.image_loaded")

        os.environ["SAM3_MASK_STUDIO_LANG"] = "ja_JP"
        self.assertEqual(i18n_helper.t("SAM3 Mask"), "SAM3 マスク")
        self.assertEqual(i18n_helper.t("preset.full_body"), "人物全体")
        self.assertEqual(
            i18n_helper.t("status.image_loaded", width=512, height=768),
            "画像を読み込みました (512x768)",
        )

    def test_mask_ops_set_operations_and_grow(self):
        shape = (20, 20)
        m_pos = mask_ops.empty_mask(shape)
        m_pos[4:16, 4:16] = 255

        m_neg = mask_ops.empty_mask(shape)
        m_neg[8:12, 8:12] = 255

        final = mask_ops.compute_final_mask([m_pos], [m_neg], shape=shape, grow_px=0)
        self.assertEqual(final.dtype, np.uint8)
        self.assertTrue(set(np.unique(final)).issubset({0, 255}))
        self.assertEqual(int(final[5, 5]), 255)
        self.assertEqual(int(final[10, 10]), 0)

        grown = mask_ops.apply_grow_shrink(m_pos, grow_px=2)
        shrunk = mask_ops.apply_grow_shrink(m_pos, grow_px=-2)
        self.assertGreater(int((grown > 0).sum()), int((m_pos > 0).sum()))
        self.assertLess(int((shrunk > 0).sum()), int((m_pos > 0).sum()))

    def test_scale_click_coords(self):
        x, y = mask_ops.scale_click_coords(50, 100, display_size=(100, 200), orig_size=(400, 800))
        self.assertEqual((x, y), (200, 400))

    def test_ui_tab_and_image_input_callback(self):
        os.environ["SAM3_MASK_STUDIO_LANG"] = "ja_JP"
        i18n_helper.clear_cache()

        img = Image.new("RGB", (64, 48), color=(120, 130, 140))
        preview, st, active_prompts, msg = callbacks.on_image_Input_change(img, None)
        self.assertIsNotNone(preview)
        self.assertEqual(preview.size, (64, 48))
        self.assertIsNotNone(st.original_image)
        self.assertIsNotNone(st.final_mask)
        self.assertEqual(st.final_mask.shape, (48, 64))
        self.assertEqual(active_prompts.get("choices"), [])
        self.assertEqual(active_prompts.get("value"), [])
        self.assertIn("64x48", msg)

        tabs = script.on_ui_tabs()
        self.assertEqual(len(tabs), 1)
        self.assertEqual(tabs[0][1], "SAM3 Mask")
        self.assertEqual(tabs[0][2], "sam3_mask_studio_tab")

    def test_sys_modules_collision_resilience(self):
        import sys
        import types

        # Simulate another extension overwriting sys.modules['install'] and sys.modules['ui']
        fake_install = types.ModuleType("install")
        fake_ui = types.ModuleType("ui")
        fake_ui.__file__ = "/some/other/extension/ui/__init__.py"
        old_install = sys.modules.get("install")
        sys.modules["install"] = fake_install
        sys.modules["ui"] = fake_ui
        try:
            tabs = script.on_ui_tabs()
            self.assertEqual(len(tabs), 1)
            self.assertEqual(tabs[0][1], "SAM3 Mask")
        finally:
            if old_install is not None:
                sys.modules["install"] = old_install


if __name__ == "__main__":
    unittest.main()
