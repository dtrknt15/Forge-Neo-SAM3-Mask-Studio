"""Unit tests for v2 improvements: multi-point refine, instance picker, guided edge refine,
soft brush, cutout/outline previews, compressed undo, layers ZIP export, batch enhancements,
VRAM status, and img2img auto-mask helper.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

EXT_ROOT = Path(__file__).resolve().parents[1]
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

import mask_ops
import sam3_backend
import transfer
from ui import callbacks


class TestImprovementsV2(unittest.TestCase):
    def setUp(self) -> None:
        self.img = Image.new("RGB", (64, 64), (120, 140, 160))
        self.mask = np.zeros((64, 64), dtype=np.uint8)
        self.mask[16:48, 16:48] = 255

    def test_vram_status_and_extract_instances(self) -> None:
        status = sam3_backend.get_vram_status()
        self.assertIn("Model:", status)

        m1 = np.zeros((64, 64), dtype=np.float32)
        m1[10:20, 10:20] = 1.0
        m2 = np.zeros((64, 64), dtype=np.float32)
        m2[30:40, 30:40] = 1.0
        fake_state = {
            "masks": np.stack([m1, m2], axis=0),
            "scores": np.array([0.45, 0.90], dtype=np.float32),
        }
        instances = sam3_backend._extract_instances_from_state(
            fake_state, target_shape=(64, 64), score_threshold=0.40
        )
        self.assertEqual(len(instances), 2)
        # Sorted descending by score
        self.assertGreater(instances[0]["score"], instances[1]["score"])
        self.assertEqual(instances[0]["index"], 1)
        self.assertEqual(instances[1]["index"], 2)

    def test_guided_refine_edge_smooth_and_soft_brush(self) -> None:
        refined = mask_ops.apply_guided_refine(self.img, self.mask, radius=4)
        self.assertEqual(refined.shape, (64, 64))
        self.assertEqual(refined.dtype, np.uint8)
        self.assertTrue(refined.any())

        smoothed = mask_ops.apply_edge_smooth(self.mask, smooth_px=2)
        self.assertEqual(smoothed.shape, (64, 64))
        self.assertEqual(set(np.unique(smoothed).tolist()), {0, 255})

        soft = mask_ops.apply_brush_stroke(
            None,
            [(32, 32)],
            radius=12,
            erase=False,
            hardness=0.3,
            opacity=0.6,
            shape=(64, 64),
        )
        self.assertEqual(soft.shape, (64, 64))
        # Soft brush should produce intermediate grayscale values
        unique_vals = np.unique(soft)
        self.assertTrue(np.any((unique_vals > 0) & (unique_vals < 255)))

    def test_cutout_outline_and_refine_markers(self) -> None:
        cutout = mask_ops.render_overlay(self.img, self.mask, view_mode="cutout")
        self.assertEqual(cutout.size, (64, 64))

        outline = mask_ops.render_overlay(
            self.img,
            self.mask,
            view_mode="outline",
            refine_points=[(32, 32), (10, 10)],
            refine_labels=[1, 0],
        )
        self.assertEqual(outline.size, (64, 64))

    def test_export_layers_zip(self) -> None:
        pos_masks = {"person": self.mask}
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path, payload_json, msg = transfer.export_layers_zip(
                self.img,
                self.mask,
                pos_prompt_masks=pos_masks,
                save_dir=tmpdir,
            )
            self.assertTrue(zip_path.is_file())
            payload = json.loads(payload_json)
            self.assertEqual(payload["mime"], "application/zip")
            raw_zip = base64.b64decode(payload["image"])
            with zipfile.ZipFile(io.BytesIO(raw_zip), "r") as zf:
                names = zf.namelist()
                self.assertIn("final_mask.png", names)
                self.assertIn("final_cutout_rgba.png", names)
                self.assertIn("pos_01_person_mask.png", names)
            self.assertIn(zip_path.name, msg)

    def test_compressed_history_and_multipoint_refine(self) -> None:
        st = callbacks.MaskStudioState()
        st.reset_for_new_image(self.img)

        st, _ = callbacks.on_refine_mode_change(True, st)
        self.assertTrue(st.refine_mode)

        def fake_point(image, point_coords, point_labels=None, **_kwargs):
            out = np.zeros((64, 64), dtype=np.uint8)
            for (px, py), lbl in zip(point_coords, point_labels or [1] * len(point_coords)):
                if lbl == 1:
                    out[max(0, py - 5) : min(64, py + 5), max(0, px - 5) : min(64, px + 5)] = 255
                else:
                    out[max(0, py - 2) : min(64, py + 2), max(0, px - 2) : min(64, px + 2)] = 0
            return out

        with mock.patch("sam3_backend.predict_point_mask", side_effect=fake_point):
            _, st, _ = callbacks.on_preview_click("Click Mode: Add", st, coords=(32, 32))
            self.assertEqual(len(st.refine_points), 1)
            self.assertEqual(st.refine_labels, [1])
            _, st, _ = callbacks.on_preview_click("Click Mode: Exclude", st, coords=(32, 32))
            self.assertEqual(len(st.refine_points), 2)
            self.assertEqual(st.refine_labels, [1, 0])

            # Remove last refine point
            _, st, _ = callbacks.on_remove_last_refine_point(st)
            self.assertEqual(len(st.refine_points), 1)

            # Commit object
            _, st, _ = callbacks.on_commit_refine_object(st)
            self.assertEqual(len(st.refine_points), 0)
            self.assertEqual(len(st.pos_click_masks), 1)

        # Undo should restore from zlib-compressed HistorySnapshot
        _, st, _ = callbacks.on_undo(st)
        self.assertEqual(len(st.refine_points), 1)
        self.assertEqual(len(st.pos_click_masks), 0)

    def test_instance_picker_and_soft_brush_callback(self) -> None:
        st = callbacks.MaskStudioState()
        st.reset_for_new_image(self.img)

        m1 = np.zeros((64, 64), dtype=np.uint8)
        m1[5:15, 5:15] = 255
        m2 = np.zeros((64, 64), dtype=np.uint8)
        m2[40:55, 40:55] = 255
        st.positive_prompts = ["person"]
        st.pos_prompt_instances["person"] = [
            {"index": 1, "score": 0.95, "mask": m1},
            {"index": 2, "score": 0.80, "mask": m2},
        ]
        st.pos_prompt_masks["person"] = mask_ops.union_masks([m1, m2], (64, 64))

        choices, selected = callbacks.format_instance_choices(st)
        self.assertEqual(len(choices), 2)
        self.assertEqual(selected, ["person#1", "person#2"])

        # Exclude instance #2
        _, st, _ = callbacks.on_instances_change(["person#1"], st)
        self.assertEqual(st.excluded_instances, {"person#2"})
        self.assertEqual(st.final_mask[10, 10], 255)
        self.assertEqual(st.final_mask[45, 45], 0)

        # Soft brush stroke
        payload = json.dumps({"points": [[32, 32]], "radius": 8, "erase": False, "width": 64, "height": 64})
        _, st, _, _, _ = callbacks.on_brush_stroke(
            payload, "Brush: Add", 8, st, brush_hardness=0.3, brush_opacity=0.5
        )
        self.assertIsNotNone(st.soft_base_mask)
        out_mask = callbacks.get_output_mask(st)
        self.assertTrue(np.any((out_mask > 0) & (out_mask < 255)))

        # Edge refine & smooth callbacks
        _, st, _ = callbacks.on_edge_refine_change(4, st)
        self.assertEqual(st.edge_refine_radius, 4)
        _, st, _ = callbacks.on_edge_smooth_change(2, st)
        self.assertEqual(st.edge_smooth_px, 2)

    def test_batch_recursive_resume_json_and_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = Path(tmpdir) / "in"
            sub_dir = in_dir / "sub"
            out_dir = Path(tmpdir) / "out"
            sub_dir.mkdir(parents=True)
            self.img.save(in_dir / "a.png")
            self.img.save(sub_dir / "b.png")

            def fake_text(_img, _prompt, **_kwargs):
                return self.mask.copy()

            with mock.patch("sam3_backend.predict_text_mask", side_effect=fake_text):
                msg1 = callbacks.on_batch_process(
                    str(in_dir),
                    str(out_dir),
                    "person",
                    "",
                    None,
                    0,
                    0,
                    0.35,
                    "Full Image",
                    "Transparent",
                    recursive=True,
                    resume_skip_existing=True,
                    save_json_metadata=True,
                    progress=None,
                )
                self.assertIn("2", msg1)
                self.assertTrue((out_dir / "a.png").is_file())
                self.assertTrue((out_dir / "a.json").is_file())
                self.assertTrue((out_dir / "sub" / "b.png").is_file())
                self.assertTrue((out_dir / "sub" / "b.json").is_file())

                # Second run with resume_skip_existing=True should skip both
                msg2 = callbacks.on_batch_process(
                    str(in_dir),
                    str(out_dir),
                    "person",
                    "",
                    None,
                    0,
                    0,
                    0.35,
                    "Full Image",
                    "Transparent",
                    recursive=True,
                    resume_skip_existing=True,
                    progress=None,
                )
                self.assertIn("2", msg2)

    def test_img2img_auto_mask_helper(self) -> None:
        import scripts.sam3_mask_studio as sms

        with mock.patch("sam3_backend.predict_text_mask", return_value=self.mask.copy()):
            out_l = sms.generate_auto_mask(self.img, "person", feather_px=2)
            self.assertIsNotNone(out_l)
            self.assertEqual(out_l.mode, "L")
            self.assertEqual(out_l.size, (64, 64))


if __name__ == "__main__":
    unittest.main()
