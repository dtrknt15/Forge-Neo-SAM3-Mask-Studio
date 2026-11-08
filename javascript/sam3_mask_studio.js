/**
 * Client-side helpers for SAM3 Mask Studio:
 * 1. Switching directly to img2img -> Inpaint upload (tab index 4)
 * 2. Fallback file input injection into #img_inpaint_base / #img_inpaint_mask
 * 3. Triggering browser file download for exported masked RGBA PNG
 * 4. Grabbing active image from img2img Forge Canvas
 * 5. Interactive Preview Image (pure event delegation, no MutationObserver):
 *    - Mouse Wheel Zoom (1.0x - 8.0x) centered on cursor
 *    - Drag to Pan (when zoomed in or with middle mouse button)
 *    - Persistent zoom/pan across mask preview updates (via capture-phase 'load')
 *    - Right-Click directly on preview image to Exclude
 */
(function () {
  function getRoot() {
    return typeof gradioApp === "function" ? gradioApp() : document;
  }

  function b64ToBlob(b64) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
      arr[i] = bin.charCodeAt(i);
    }
    return new Blob([arr], { type: "image/png" });
  }

  function setFileInput(container, file) {
    if (!container) return false;
    const input = container.querySelector("input[type=file]");
    if (!input) return false;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  function grabImg2ImgCanvas() {
    try {
      const root = getRoot();
      const EXCLUDE =
        /controlnet|control_net|control-net|alwayson|t2i_adapter|ip[_-]?adapter|reference_only|animatediff|reactor|adetailer/i;

      function blobOf(n) {
        const id = n.id || "";
        const cls = typeof n.className === "string" ? n.className : "";
        const label = (n.getAttribute && n.getAttribute("aria-label")) || "";
        return (id + " " + cls + " " + label).toLowerCase();
      }

      function excluded(el) {
        let n = el;
        while (n && n !== root) {
          if (EXCLUDE.test(blobOf(n))) return true;
          n = n.parentElement;
        }
        return false;
      }

      function inImg2Img(el) {
        let n = el;
        let saw = false;
        while (n && n !== root) {
          const id = (n.id || "").toLowerCase();
          if (id.indexOf("txt2img") !== -1 || id.indexOf("sam3_ms") !== -1) return false;
          if (id.indexOf("img_inpaint") === 0) return true;
          if (id.indexOf("img2img") !== -1 || id === "mode_img2img") saw = true;
          n = n.parentElement;
        }
        return saw;
      }

      const areas = root.querySelectorAll(".logical_image_background textarea");
      for (const ta of areas) {
        if (!ta.value || ta.value.indexOf("data:image/") !== 0) continue;
        if (excluded(ta)) continue;
        if (!inImg2Img(ta)) continue;
        return ta.value;
      }
    } catch (e) {}
    return "";
  }

  function focusInpaintUploadTab() {
    const root = getRoot();

    function doSwitch() {
      if (typeof switch_to_img2img_tab === "function") {
        try {
          switch_to_img2img_tab(4);
          return true;
        } catch (e) {}
      }

      try {
        const topTabs = root.querySelector("#tabs");
        if (topTabs) {
          const topBtns = topTabs.querySelectorAll("button");
          if (topBtns.length > 1) topBtns[1].click();
        }
      } catch (e) {}

      try {
        const mode = root.querySelector("#mode_img2img");
        if (mode) {
          const modeBtns = mode.querySelectorAll("button");
          if (modeBtns.length > 4) {
            modeBtns[4].click();
            return true;
          }
        }
      } catch (e) {}

      return false;
    }

    doSwitch();
    requestAnimationFrame(function () {
      try {
        doSwitch();
      } catch (e) {}
    });

    try {
      window.scrollTo(0, 0);
    } catch (e) {}

    return Array.from(arguments);
  }

  function injectInpaintUpload(payload) {
    focusInpaintUploadTab();
    if (!payload) return "no_payload";
    let data = payload;
    if (typeof payload === "string") {
      try {
        data = JSON.parse(payload);
      } catch (e) {
        return "bad_json";
      }
    }
    const imgB64 = data && data.image;
    const maskB64 = data && data.mask;
    if (!imgB64 || !maskB64) return "missing_b64";

    const root = getRoot();
    setTimeout(() => {
      const baseEl = root.querySelector("#img_inpaint_base");
      const maskEl = root.querySelector("#img_inpaint_mask");
      if (baseEl && maskEl) {
        setFileInput(
          baseEl,
          new File([b64ToBlob(imgB64)], "sam3_inpaint_init.png", { type: "image/png" })
        );
        setFileInput(
          maskEl,
          new File([b64ToBlob(maskB64)], "sam3_inpaint_mask.png", { type: "image/png" })
        );
      }
    }, 100);
    return "ok";
  }

  function downloadMaskedPng(payload) {
    if (!payload) return "no_payload";
    let data = payload;
    if (typeof payload === "string") {
      try {
        data = JSON.parse(payload);
      } catch (e) {
        return "bad_json";
      }
    }
    const b64 = data && data.image;
    const filename = (data && data.filename) || "sam3_masked.png";
    if (!b64) return "missing_b64";

    const blob = b64ToBlob(b64);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    return "ok";
  }

  // =========================================================================
  // Interactive Preview Controller (Pure Event Delegation, No MutationObserver)
  // =========================================================================
  const zoomState = {
    scale: 1,
    tx: 0,
    ty: 0,
    naturalW: 0,
    naturalH: 0,
    dragging: false,
    startX: 0,
    startY: 0,
    startTx: 0,
    startTy: 0,
    movedDist: 0,
    suppressNextClick: false,
  };

  function getPreviewContainer() {
    const root = getRoot();
    return root ? root.querySelector("#sam3_ms_preview_image") : null;
  }

  function getPreviewImg() {
    const container = getPreviewContainer();
    return container ? container.querySelector("img") : null;
  }

  function insidePreview(target) {
    if (!target || !(target instanceof Element)) return false;
    return Boolean(target.closest("#sam3_ms_preview_image"));
  }

  function ensureZoomBadge(container) {
    let badge = container.querySelector(".sam3-ms-zoom-badge");
    if (!badge) {
      badge = document.createElement("div");
      badge.className = "sam3-ms-zoom-badge";
      badge.title = "Click to reset zoom (100%)";
      badge.addEventListener("click", function (e) {
        e.stopPropagation();
        e.preventDefault();
        resetZoom();
      });
      container.appendChild(badge);
    }
    return badge;
  }

  function clampPan() {
    if (zoomState.scale <= 1.001) {
      zoomState.scale = 1;
      zoomState.tx = 0;
      zoomState.ty = 0;
      return;
    }
    const img = getPreviewImg();
    if (!img) return;
    const w = img.clientWidth || 400;
    const h = img.clientHeight || 400;
    const minTx = w * (1 - zoomState.scale);
    const minTy = h * (1 - zoomState.scale);
    zoomState.tx = Math.min(0, Math.max(minTx, zoomState.tx));
    zoomState.ty = Math.min(0, Math.max(minTy, zoomState.ty));
  }

  function applyTransform() {
    const container = getPreviewContainer();
    const img = getPreviewImg();
    if (!container || !img) return;

    clampPan();
    img.style.transformOrigin = "0 0";
    if (zoomState.scale <= 1.001) {
      if (img.style.transform !== "") img.style.transform = "";
      img.style.cursor = "crosshair";
    } else {
      const nextTransform = `translate(${zoomState.tx}px, ${zoomState.ty}px) scale(${zoomState.scale})`;
      if (img.style.transform !== nextTransform) {
        img.style.transform = nextTransform;
      }
      img.style.cursor = zoomState.dragging ? "grabbing" : "crosshair";
    }

    const badge = ensureZoomBadge(container);
    const pct = Math.round(zoomState.scale * 100);
    const label = `${pct}%`;
    if (badge.textContent !== label) {
      badge.textContent = label;
    }
    const disp = pct > 100 ? "block" : "none";
    if (badge.style.display !== disp) {
      badge.style.display = disp;
    }
  }

  function resetZoom() {
    zoomState.scale = 1;
    zoomState.tx = 0;
    zoomState.ty = 0;
    applyTransform();
  }

  function installDelegatedListeners() {
    if (window._sam3MaskStudioDelegatedInstalled) return;
    window._sam3MaskStudioDelegatedInstalled = true;

    document.addEventListener(
      "load",
      function (e) {
        const t = e.target;
        if (!t || t.tagName !== "IMG" || !insidePreview(t)) return;
        if (
          zoomState.naturalW &&
          zoomState.naturalH &&
          (t.naturalWidth !== zoomState.naturalW || t.naturalHeight !== zoomState.naturalH)
        ) {
          zoomState.scale = 1;
          zoomState.tx = 0;
          zoomState.ty = 0;
        }
        zoomState.naturalW = t.naturalWidth;
        zoomState.naturalH = t.naturalHeight;
        applyTransform();
      },
      true
    );

    // 1. Mouse Wheel Zoom centered at cursor
    document.addEventListener(
      "wheel",
      function (e) {
        if (!insidePreview(e.target)) return;
        const currentImg = getPreviewImg();
        if (!currentImg) return;
        e.preventDefault();
        e.stopPropagation();

        const rect = currentImg.getBoundingClientRect();
        const prevScale = zoomState.scale;
        const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
        const nextScale = Math.min(8.0, Math.max(1.0, prevScale * factor));
        if (Math.abs(nextScale - prevScale) < 1e-4) return;

        const offsetX = e.clientX - rect.left;
        const offsetY = e.clientY - rect.top;
        const ratio = nextScale / prevScale;

        zoomState.tx = zoomState.tx - offsetX * (ratio - 1);
        zoomState.ty = zoomState.ty - offsetY * (ratio - 1);
        zoomState.scale = nextScale;
        applyTransform();
      },
      { passive: false }
    );

    // 2. Drag to Pan (Left-button when zoomed in, or Middle-button anytime)
    document.addEventListener("mousedown", function (e) {
      if (!insidePreview(e.target)) return;
      const currentImg = getPreviewImg();
      if (!currentImg) return;
      if (e.button === 1 || (e.button === 0 && zoomState.scale > 1.001)) {
        zoomState.dragging = true;
        zoomState.startX = e.clientX;
        zoomState.startY = e.clientY;
        zoomState.startTx = zoomState.tx;
        zoomState.startTy = zoomState.ty;
        zoomState.movedDist = 0;
        if (e.button === 1) {
          e.preventDefault();
        }
      }
    });

    window.addEventListener("mousemove", function (e) {
      if (!zoomState.dragging) return;
      const dx = e.clientX - zoomState.startX;
      const dy = e.clientY - zoomState.startY;
      zoomState.movedDist = Math.hypot(dx, dy);
      if (zoomState.movedDist > 4) {
        zoomState.tx = zoomState.startTx + dx;
        zoomState.ty = zoomState.startTy + dy;
        applyTransform();
      }
    });

    window.addEventListener("mouseup", function () {
      if (!zoomState.dragging) return;
      if (zoomState.movedDist > 5) {
        zoomState.suppressNextClick = true;
        setTimeout(() => {
          zoomState.suppressNextClick = false;
        }, 80);
      }
      zoomState.dragging = false;
      applyTransform();
    });

    // Suppress Gradio .select() click if the user just dragged to pan
    document.addEventListener(
      "click",
      function (e) {
        if (!insidePreview(e.target)) return;
        if (zoomState.suppressNextClick && !e._sam3SyntheticRightClick) {
          zoomState.suppressNextClick = false;
          e.stopPropagation();
          e.preventDefault();
        }
      },
      true
    );

    // Double-click resets zoom to 100%
    document.addEventListener("dblclick", function (e) {
      if (!insidePreview(e.target)) return;
      if (zoomState.scale > 1.001) {
        e.preventDefault();
        e.stopPropagation();
        resetZoom();
      }
    });

    // 3. Right-Click on Preview Image -> Exclude at that coordinate
    document.addEventListener("contextmenu", function (e) {
      if (!insidePreview(e.target)) return;
      const currentImg = getPreviewImg();
      if (!currentImg || !currentImg.contains(e.target)) return;
      e.preventDefault();
      e.stopPropagation();

      const cx = e.clientX;
      const cy = e.clientY;
      const root = getRoot();
      const modeContainer = root ? root.querySelector("#sam3_ms_click_mode") : null;
      const radios = modeContainer
        ? modeContainer.querySelectorAll('input[type="radio"]')
        : [];
      const addRadio = radios.length > 0 ? radios[0] : null;
      const excludeRadio = radios.length > 1 ? radios[1] : null;
      const wasAddChecked = addRadio ? addRadio.checked : false;

      if (excludeRadio && !excludeRadio.checked) {
        excludeRadio.click();
      }

      setTimeout(function () {
        const targetImg = getPreviewImg();
        if (!targetImg) return;
        const clickEvt = new MouseEvent("click", {
          bubbles: true,
          cancelable: true,
          clientX: cx,
          clientY: cy,
        });
        clickEvt._sam3SyntheticRightClick = true;
        targetImg.dispatchEvent(clickEvt);

        if (wasAddChecked && addRadio) {
          setTimeout(function () {
            addRadio.click();
          }, 160);
        }
      }, 35);
    });
  }

  if (typeof window !== "undefined") {
    window.sam3MaskStudioFocusInpaintUpload = focusInpaintUploadTab;
    window.sam3MaskStudioInjectInpaint = injectInpaintUpload;
    window.sam3MaskStudioDownloadPng = downloadMaskedPng;
    window.sam3MaskStudioResetZoom = resetZoom;
    window.sam3MaskStudioGrabImg2Img = grabImg2ImgCanvas;
    installDelegatedListeners();
  }
})();
