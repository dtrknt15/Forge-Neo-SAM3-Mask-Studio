"""Standalone 'SAM3 Mask' tab UI builder with full M1–M6 event bindings +
Preview Toolbar (Click Mode, Right-Click Exclude, Invert Mask, Zoom/Pan,
Load from img2img, and Score Threshold).
"""

from __future__ import annotations

from typing import Any

import gradio as gr

import sam3_backend
from ui import callbacks, presets


def build_tab(
    inpaint_base_comp: Any = None,
    inpaint_mask_comp: Any = None,
) -> gr.Blocks:
    """Construct the standalone 'SAM3 Mask' Gradio Blocks tab."""
    models_dir_display = str(sam3_backend.primary_models_dir())
    available_ckpts = sam3_backend.list_checkpoints()
    model_ready = bool(available_ckpts)
    default_ckpt = available_ckpts[0] if available_ckpts else None
    preset_choices = presets.get_dropdown_choices()

    with gr.Blocks(analytics_enabled=False) as tab_blocks:
        state = gr.State(callbacks.MaskStudioState())
        img2img_canvas_b64 = gr.Textbox(value="", visible=False, elem_id="sam3_ms_img2img_b64")

        # 1. Model Setup Accordion (expanded only when model is not yet set up)
        with gr.Accordion(
            "Model Setup",
            open=not model_ready,
            elem_id="sam3_ms_setup_accordion",
        ):
            gr.Markdown(
                f"1. [Approve License on Hugging Face (facebook/sam3)](https://huggingface.co/facebook/sam3)\n"
                f"2. Manual placement directory: `{models_dir_display}`",
                elem_id="sam3_ms_setup_guide",
            )
            with gr.Row():
                hf_token_input = gr.Textbox(
                    label="Hugging Face Token (leave blank to use saved local token)",
                    type="password",
                    placeholder="hf_...",
                    scale=3,
                    elem_id="sam3_ms_hf_token",
                )
                check_access_btn = gr.Button(
                    "Check Access",
                    scale=1,
                    elem_id="sam3_ms_check_access_btn",
                )
                download_model_btn = gr.Button(
                    "Download SAM3 Model",
                    variant="primary",
                    scale=1,
                    elem_id="sam3_ms_download_btn",
                )
            with gr.Row():
                ckpt_dropdown = gr.Dropdown(
                    label="SAM3 Checkpoint",
                    choices=available_ckpts,
                    value=default_ckpt,
                    scale=3,
                    elem_id="sam3_ms_ckpt_dropdown",
                )
                load_model_btn = gr.Button(
                    "Load Model",
                    scale=1,
                    elem_id="sam3_ms_load_btn",
                )
                unload_model_btn = gr.Button(
                    "Unload / Free VRAM",
                    scale=1,
                    elem_id="sam3_ms_unload_btn",
                )
            setup_status = gr.Markdown(
                f"Status: Checkpoint detected (`{default_ckpt}`)"
                if model_ready
                else f"Status: No checkpoint found in `{models_dir_display}`",
                elem_id="sam3_ms_setup_status",
            )

        # 2. Main 2-Column Workspace (Left: Input + Text Mask | Right: Preview + Toolbar + Output)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                    height=380,
                    elem_id="sam3_ms_input_image",
                )
                with gr.Row():
                    load_img2img_btn = gr.Button(
                        "Load Image from img2img",
                        size="sm",
                        scale=1,
                        elem_id="sam3_ms_load_img2img_btn",
                    )
                    import_mask_btn = gr.UploadButton(
                        "Import Mask PNG",
                        file_types=["image"],
                        size="sm",
                        scale=1,
                        elem_id="sam3_ms_import_mask_btn",
                    )

                # Text Mask Section (in Left Column for 1-screen workflow)
                with gr.Accordion("Text Mask", open=True, elem_id="sam3_ms_text_accordion"):
                    with gr.Row():
                        pos_prompt_input = gr.Textbox(
                            label="Positive Prompt",
                            placeholder="e.g. person, hair, red dress",
                            scale=3,
                            elem_id="sam3_ms_pos_prompt",
                        )
                        pos_preset_dropdown = gr.Dropdown(
                            label="Presets",
                            choices=preset_choices,
                            value=None,
                            scale=2,
                            elem_id="sam3_ms_pos_preset",
                        )
                        add_pos_btn = gr.Button(
                            "Add Positive",
                            variant="primary",
                            scale=1,
                            elem_id="sam3_ms_add_pos_btn",
                        )
                    with gr.Row():
                        neg_prompt_input = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="e.g. face, background",
                            scale=3,
                            elem_id="sam3_ms_neg_prompt",
                        )
                        neg_preset_dropdown = gr.Dropdown(
                            label="Negative Presets",
                            choices=preset_choices,
                            value=None,
                            scale=2,
                            elem_id="sam3_ms_neg_preset",
                        )
                        add_neg_btn = gr.Button(
                            "Add Negative",
                            scale=1,
                            elem_id="sam3_ms_add_neg_btn",
                        )
                    active_prompts_group = gr.CheckboxGroup(
                        label="Applied Prompts (uncheck to remove)",
                        choices=[],
                        value=[],
                        elem_id="sam3_ms_active_prompts",
                    )
                    with gr.Accordion("Manage Custom Presets", open=False, elem_id="sam3_ms_custom_preset_acc"):
                        with gr.Row():
                            custom_preset_name = gr.Textbox(
                                label="Preset ID / Name",
                                placeholder="e.g. bangs_only",
                                scale=2,
                                elem_id="sam3_ms_custom_preset_name",
                            )
                            save_preset_btn = gr.Button(
                                "Save Current Inputs as Preset",
                                scale=1,
                                elem_id="sam3_ms_save_preset_btn",
                            )
                            delete_preset_btn = gr.Button(
                                "Delete Selected Preset",
                                scale=1,
                                elem_id="sam3_ms_delete_preset_btn",
                            )

            with gr.Column(scale=1):
                preview_image = gr.Image(
                    label="Mask Preview",
                    type="pil",
                    image_mode="RGB",
                    interactive=False,
                    height=440,
                    elem_id="sam3_ms_preview_image",
                )
                gr.Markdown(
                    "<sub>Left-Click: Add/Mode | Right-Click: Exclude | Shift+Drag: Box | Wheel: Zoom | Space+Drag: Pan</sub>",
                    elem_id="sam3_ms_preview_hint",
                )
                # Compact interactive controls directly under the preview image
                with gr.Row(elem_id="sam3_ms_preview_mode_row"):
                    click_mode_radio = gr.Radio(
                        choices=["Click Mode: Add", "Click Mode: Exclude", "Brush: Add", "Brush: Erase"],
                        value="Click Mode: Add",
                        label="Click Mode",
                        scale=3,
                        elem_id="sam3_ms_click_mode",
                    )
                    brush_size_slider = gr.Slider(
                        label="Brush Size (px)",
                        minimum=2,
                        maximum=120,
                        step=2,
                        value=20,
                        scale=1,
                        visible=False,
                        elem_id="sam3_ms_brush_size",
                    )
                with gr.Row(elem_id="sam3_ms_preview_view_row"):
                    preview_mode_radio = gr.Radio(
                        choices=["Overlay", "Mask Only", "Original"],
                        value="Overlay",
                        label="Preview View",
                        scale=2,
                        elem_id="sam3_ms_preview_mode",
                    )
                    overlay_opacity_slider = gr.Slider(
                        label="Overlay Opacity",
                        minimum=0.10,
                        maximum=0.90,
                        step=0.05,
                        value=0.45,
                        scale=1,
                        elem_id="sam3_ms_overlay_opacity",
                    )
                    color_by_prompt_chk = gr.Checkbox(
                        label="Color-code by text prompt",
                        value=False,
                        scale=1,
                        elem_id="sam3_ms_color_by_prompt",
                    )
                with gr.Row(elem_id="sam3_ms_preview_actions_row"):
                    invert_mask_btn = gr.Button(
                        "Invert Mask",
                        size="sm",
                        scale=1,
                        elem_id="sam3_ms_invert_btn",
                    )
                    undo_btn = gr.Button(
                        "Undo (Ctrl+Z)",
                        size="sm",
                        scale=1,
                        elem_id="sam3_ms_undo_btn",
                    )
                    clear_mask_btn = gr.Button(
                        "Clear Mask",
                        size="sm",
                        scale=1,
                        elem_id="sam3_ms_clear_btn",
                    )
                    reset_zoom_btn = gr.Button(
                        "Reset Zoom",
                        size="sm",
                        scale=1,
                        elem_id="sam3_ms_reset_zoom_btn",
                    )

                # Output Section (in Right Column for immediate access without scrolling)
                with gr.Accordion("Output", open=True, elem_id="sam3_ms_output_accordion"):
                    with gr.Row():
                        send_inpaint_btn = gr.Button(
                            "Send to Inpaint upload",
                            variant="primary",
                            elem_id="sam3_ms_send_inpaint_btn",
                        )
                        download_png_btn = gr.Button(
                            "Download Masked PNG",
                            elem_id="sam3_ms_download_png_btn",
                        )
                    with gr.Row():
                        crop_mode_radio = gr.Radio(
                            choices=["Full Image", "Crop to Mask"],
                            value="Full Image",
                            label="Region Option",
                            elem_id="sam3_ms_crop_mode",
                        )
                        bg_mode_radio = gr.Radio(
                            choices=["Transparent", "White Background", "Mask Only (B&W)"],
                            value="Transparent",
                            label="Background Option",
                            elem_id="sam3_ms_bg_mode",
                        )

        status_bar = gr.Textbox(
            label="Status",
            value="Ready. Upload an image to start.",
            interactive=False,
            max_lines=1,
            elem_id="sam3_ms_status_bar",
        )

        # 3. Mask Adjustment Section
        with gr.Accordion("Mask Adjustment", open=True, elem_id="sam3_ms_adjust_accordion"):
            with gr.Row():
                grow_slider = gr.Slider(
                    label="Grow / Shrink (px)",
                    minimum=-64,
                    maximum=64,
                    step=1,
                    value=0,
                    scale=1,
                    elem_id="sam3_ms_grow_slider",
                )
                feather_slider = gr.Slider(
                    label="Feather / Blur (px)",
                    minimum=0,
                    maximum=64,
                    step=1,
                    value=0,
                    scale=1,
                    elem_id="sam3_ms_feather_slider",
                )
                score_thr_slider = gr.Slider(
                    label="Detection Score Threshold",
                    minimum=0.05,
                    maximum=0.95,
                    step=0.05,
                    value=callbacks.DEFAULT_SCORE_THRESHOLD,
                    scale=1,
                    elem_id="sam3_ms_score_thr_slider",
                )
            with gr.Row():
                fill_holes_btn = gr.Button(
                    "Fill Holes",
                    size="sm",
                    scale=1,
                    elem_id="sam3_ms_fill_holes_btn",
                )
                min_island_area = gr.Number(
                    label="Min Island Area (px)",
                    value=64,
                    minimum=1,
                    maximum=100000,
                    precision=0,
                    scale=1,
                    elem_id="sam3_ms_min_island_area",
                )
                remove_islands_btn = gr.Button(
                    "Remove Small Islands",
                    size="sm",
                    scale=1,
                    elem_id="sam3_ms_remove_islands_btn",
                )

        inpaint_payload_box = gr.Textbox(visible=False, elem_id="sam3_ms_inpaint_payload")
        download_payload_box = gr.Textbox(visible=False, elem_id="sam3_ms_download_payload")
        box_payload_box = gr.Textbox(value="", visible=False, elem_id="sam3_ms_box_payload")
        brush_payload_box = gr.Textbox(value="", visible=False, elem_id="sam3_ms_brush_payload")

        # 6. Batch Processing Section
        with gr.Accordion("Batch Processing", open=False, elem_id="sam3_ms_batch_accordion"):
            with gr.Row():
                batch_input_dir = gr.Textbox(
                    label="Input Directory",
                    placeholder="C:\\path\\to\\images",
                    scale=2,
                    elem_id="sam3_ms_batch_in_dir",
                )
                batch_output_dir = gr.Textbox(
                    label="Output Directory (optional)",
                    placeholder="Default: <Input>/sam3_masks",
                    scale=2,
                    elem_id="sam3_ms_batch_out_dir",
                )
            with gr.Row():
                batch_preset_dropdown = gr.Dropdown(
                    label="Batch Preset",
                    choices=preset_choices,
                    value=None,
                    scale=1,
                    elem_id="sam3_ms_batch_preset",
                )
                batch_pos_prompt = gr.Textbox(
                    label="Batch Positive Prompt",
                    placeholder="e.g. person, hair",
                    scale=2,
                    elem_id="sam3_ms_batch_pos",
                )
                batch_neg_prompt = gr.Textbox(
                    label="Batch Negative Prompt",
                    placeholder="e.g. background",
                    scale=2,
                    elem_id="sam3_ms_batch_neg",
                )
                batch_run_btn = gr.Button(
                    "Run Batch Masking",
                    variant="primary",
                    scale=1,
                    elem_id="sam3_ms_batch_run_btn",
                )

        # ===================================================================
        # Event Bindings (M1 - M6)
        # ===================================================================

        # M1: Image Input, Load from img2img & Color-coding
        input_image.change(
            fn=callbacks.on_image_Input_change,
            inputs=[input_image, state],
            outputs=[preview_image, state, active_prompts_group, status_bar],
        )

        def _load_img2img_wrapper(canvas_b64: str, st: callbacks.MaskStudioState):
            return callbacks.on_load_from_img2img(
                canvas_b64, st, inpaint_base_comp=inpaint_base_comp
            )

        grab_event = load_img2img_btn.click(
            fn=lambda x: x,
            inputs=[img2img_canvas_b64],
            outputs=[img2img_canvas_b64],
            js=(
                "(c) => {"
                " if (window.sam3MaskStudioGrabImg2Img) {"
                "   return [window.sam3MaskStudioGrabImg2Img()];"
                " }"
                " return [c || ''];"
                "}"
            ),
        )
        grab_event.then(
            fn=_load_img2img_wrapper,
            inputs=[img2img_canvas_b64, state],
            outputs=[input_image, preview_image, state, active_prompts_group, status_bar],
        )
        import_mask_btn.upload(
            fn=callbacks.on_import_base_mask,
            inputs=[import_mask_btn, state],
            outputs=[preview_image, state, status_bar],
        )

        color_by_prompt_chk.change(
            fn=callbacks.on_color_mode_change,
            inputs=[color_by_prompt_chk, state],
            outputs=[preview_image, state],
        )
        preview_mode_radio.change(
            fn=callbacks.on_preview_mode_change,
            inputs=[preview_mode_radio, state],
            outputs=[preview_image, state],
        )
        overlay_opacity_slider.release(
            fn=callbacks.on_overlay_opacity_change,
            inputs=[overlay_opacity_slider, state],
            outputs=[preview_image, state],
        )

        # M2: Text Mask & Presets
        add_pos_btn.click(
            fn=callbacks.on_add_positive,
            inputs=[pos_prompt_input, pos_preset_dropdown, state, ckpt_dropdown],
            outputs=[preview_image, state, active_prompts_group, pos_prompt_input, status_bar],
        )
        pos_prompt_input.submit(
            fn=callbacks.on_add_positive,
            inputs=[pos_prompt_input, pos_preset_dropdown, state, ckpt_dropdown],
            outputs=[preview_image, state, active_prompts_group, pos_prompt_input, status_bar],
        )
        add_neg_btn.click(
            fn=callbacks.on_add_negative,
            inputs=[neg_prompt_input, neg_preset_dropdown, state, ckpt_dropdown],
            outputs=[preview_image, state, active_prompts_group, neg_prompt_input, status_bar],
        )
        neg_prompt_input.submit(
            fn=callbacks.on_add_negative,
            inputs=[neg_prompt_input, neg_preset_dropdown, state, ckpt_dropdown],
            outputs=[preview_image, state, active_prompts_group, neg_prompt_input, status_bar],
        )
        active_prompts_group.input(
            fn=callbacks.on_active_prompts_change,
            inputs=[active_prompts_group, state],
            outputs=[preview_image, state, active_prompts_group, status_bar],
        )
        save_preset_btn.click(
            fn=callbacks.on_save_custom_preset,
            inputs=[custom_preset_name, pos_prompt_input, neg_prompt_input],
            outputs=[pos_preset_dropdown, neg_preset_dropdown, status_bar],
        )
        delete_preset_btn.click(
            fn=callbacks.on_delete_custom_preset,
            inputs=[pos_preset_dropdown],
            outputs=[pos_preset_dropdown, neg_preset_dropdown, status_bar],
        )

        # M3: Click Operations on Preview Image
        def _handle_preview_select(
            evt: gr.SelectData,
            mode: str,
            st: callbacks.MaskStudioState,
            ckpt_name: str | None,
        ):
            return callbacks.on_preview_click(mode, st, evt=evt, ckpt_name=ckpt_name)

        preview_image.select(
            fn=_handle_preview_select,
            inputs=[click_mode_radio, state, ckpt_dropdown],
            outputs=[preview_image, state, status_bar],
        )
        box_payload_box.change(
            fn=callbacks.on_box_select,
            inputs=[box_payload_box, click_mode_radio, state, ckpt_dropdown],
            outputs=[preview_image, state, status_bar],
        )
        brush_payload_box.change(
            fn=callbacks.on_brush_stroke,
            inputs=[brush_payload_box, click_mode_radio, brush_size_slider, state],
            outputs=[preview_image, state, active_prompts_group, grow_slider, status_bar],
        )

        def _toggle_brush_slider(mode: str):
            is_brush = "Brush" in (mode or "") or "ブラシ" in (mode or "") or "消しゴム" in (mode or "")
            return gr.update(visible=is_brush)

        click_mode_radio.change(
            fn=_toggle_brush_slider,
            inputs=[click_mode_radio],
            outputs=[brush_size_slider],
        )

        # M4: Mask Adjustment (Grow / Shrink, Score Threshold, Invert, Clear, Undo, Reset Zoom)
        grow_slider.release(
            fn=callbacks.on_grow_change,
            inputs=[grow_slider, state],
            outputs=[preview_image, state, status_bar],
        )
        feather_slider.release(
            fn=callbacks.on_feather_change,
            inputs=[feather_slider, state],
            outputs=[preview_image, state, status_bar],
        )
        fill_holes_btn.click(
            fn=callbacks.on_fill_holes,
            inputs=[state],
            outputs=[preview_image, state, active_prompts_group, grow_slider, status_bar],
        )
        remove_islands_btn.click(
            fn=callbacks.on_remove_small_islands,
            inputs=[min_island_area, state],
            outputs=[preview_image, state, active_prompts_group, grow_slider, status_bar],
        )
        score_thr_slider.release(
            fn=callbacks.on_score_threshold_change,
            inputs=[score_thr_slider, state, ckpt_dropdown],
            outputs=[preview_image, state, status_bar],
        )
        invert_mask_btn.click(
            fn=callbacks.on_invert_mask,
            inputs=[state],
            outputs=[preview_image, state, active_prompts_group, grow_slider, status_bar],
        )

        def _clear_ui_wrapper(st: callbacks.MaskStudioState):
            prev, new_st, chips, grow_val, msg = callbacks.on_clear_mask(st)
            return prev, new_st, chips, grow_val, new_st.feather_px, msg

        clear_mask_btn.click(
            fn=_clear_ui_wrapper,
            inputs=[state],
            outputs=[preview_image, state, active_prompts_group, grow_slider, feather_slider, status_bar],
        )

        def _undo_ui_wrapper(st: callbacks.MaskStudioState):
            prev, new_st, chips, grow_val, thr_val, msg = callbacks.on_undo_with_ui(st)
            return prev, new_st, chips, grow_val, new_st.feather_px, thr_val, msg

        undo_btn.click(
            fn=_undo_ui_wrapper,
            inputs=[state],
            outputs=[
                preview_image,
                state,
                active_prompts_group,
                grow_slider,
                feather_slider,
                score_thr_slider,
                status_bar,
            ],
        )
        reset_zoom_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            js=(
                "() => {"
                " if (window.sam3MaskStudioResetZoom) {"
                "   window.sam3MaskStudioResetZoom();"
                " }"
                " return [];"
                "}"
            ),
        )

        # M5: Output (Send to Inpaint upload & Download Masked PNG)
        has_inpaint = inpaint_base_comp is not None and inpaint_mask_comp is not None
        inpaint_outputs: list[Any] = [status_bar, inpaint_payload_box]
        if has_inpaint:
            inpaint_outputs.extend([inpaint_base_comp, inpaint_mask_comp])

        def _send_inpaint_wrapper(st: callbacks.MaskStudioState):
            return callbacks.on_send_to_inpaint(st, has_inpaint_components=has_inpaint)

        send_event = send_inpaint_btn.click(
            fn=_send_inpaint_wrapper,
            inputs=[state],
            outputs=inpaint_outputs,
        )
        if has_inpaint:
            send_event.then(
                fn=None,
                inputs=[],
                outputs=[],
                js=(
                    "() => {"
                    " if (window.sam3MaskStudioFocusInpaintUpload) {"
                    "   window.sam3MaskStudioFocusInpaintUpload();"
                    " }"
                    " return [];"
                    "}"
                ),
            )
        else:
            send_event.then(
                fn=None,
                inputs=[inpaint_payload_box],
                outputs=[],
                js=(
                    "(payload) => {"
                    " if (window.sam3MaskStudioInjectInpaint && payload) {"
                    "   window.sam3MaskStudioInjectInpaint(payload);"
                    " } else if (window.sam3MaskStudioFocusInpaintUpload) {"
                    "   window.sam3MaskStudioFocusInpaintUpload();"
                    " }"
                    " return [];"
                    "}"
                ),
            )

        dl_event = download_png_btn.click(
            fn=callbacks.on_download_masked_png,
            inputs=[crop_mode_radio, bg_mode_radio, state],
            outputs=[status_bar, download_payload_box],
        )
        dl_event.then(
            fn=None,
            inputs=[download_payload_box],
            outputs=[],
            js=(
                "(payload) => {"
                " if (window.sam3MaskStudioDownloadPng && payload) {"
                "   window.sam3MaskStudioDownloadPng(payload);"
                " }"
                " return [];"
                "}"
            ),
        )

        # M6: Model Setup & VRAM Controls
        check_access_btn.click(
            fn=callbacks.on_check_hf_access,
            inputs=[hf_token_input],
            outputs=[setup_status],
        )
        download_model_btn.click(
            fn=callbacks.on_download_sam3_model,
            inputs=[hf_token_input],
            outputs=[setup_status, ckpt_dropdown],
        )
        load_model_btn.click(
            fn=callbacks.on_load_model,
            inputs=[ckpt_dropdown],
            outputs=[status_bar],
        )
        batch_run_btn.click(
            fn=callbacks.on_batch_process,
            inputs=[
                batch_input_dir,
                batch_output_dir,
                batch_pos_prompt,
                batch_neg_prompt,
                batch_preset_dropdown,
                grow_slider,
                feather_slider,
                score_thr_slider,
                crop_mode_radio,
                bg_mode_radio,
                ckpt_dropdown,
            ],
            outputs=[status_bar],
        )
        unload_model_btn.click(
            fn=callbacks.on_unload_model,
            inputs=[],
            outputs=[status_bar],
        )

    return tab_blocks
