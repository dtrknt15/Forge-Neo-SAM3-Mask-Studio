/**
 * Client-side helpers for SAM3 Mask Studio:
 * 1. Switching directly to img2img -> Inpaint upload (tab index 4)
 * 2. Fallback file input injection into #img_inpaint_base / #img_inpaint_mask
 * 3. Triggering browser file download for exported masked RGBA PNG
 * 4. Grabbing active image from img2img Forge Canvas
 * 5. Interactive Preview Image (pure event delegation, no MutationObserver):
 *    - Mouse Wheel Zoom (1.0x - 8.0x) centered on cursor
 *    - Native browser <img> dragstart blocked so Drag-to-Pan never misfires as a click
 *    - Space + Drag / Middle-Drag / Zoomed Left-Drag to Pan with strict click suppression
 *    - Right-Click directly on preview image to Exclude
 *    - Ctrl + Z keyboard shortcut for Undo when the SAM3 Mask tab is active
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
  // Interactive Preview Controller (Zoom, Pan, Right-Click Exclude, Ctrl+Z)
  // =========================================================================
  const PAN_THRESHOLD_PX = 3;

  const zoomState = {
    scale: 1,
    tx: 0,
    ty: 0,
    naturalW: 0,
    naturalH: 0,
    dragging: false,
    spaceHeld: false,
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

  function isTypingTarget(el) {
    if (!el || !(el instanceof Element)) return false;
    const tag = (el.tagName || "").toUpperCase();
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
      const type = (el.getAttribute("type") || "").toLowerCase();
      // Radio and checkbox inputs are not text entry fields
      if (type === "radio" || type === "checkbox" || type === "button") {
        return false;
      }
      return true;
    }
    return Boolean(el.isContentEditable);
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

  function updateCursor(img) {
    if (!img) return;
    if (zoomState.dragging) {
      img.style.cursor = "grabbing";
    } else if (zoomState.spaceHeld) {
      img.style.cursor = "grab";
    } else {
      img.style.cursor = "crosshair";
    }
  }

  function applyTransform() {
    const container = getPreviewContainer();
    const img = getPreviewImg();
    if (!container || !img) return;

    img.setAttribute("draggable", "false");
    clampPan();
    img.style.transformOrigin = "0 0";
    if (zoomState.scale <= 1.001) {
      if (img.style.transform !== "") img.style.transform = "";
    } else {
      const nextTransform = `translate(${zoomState.tx}px, ${zoomState.ty}px) scale(${zoomState.scale})`;
      if (img.style.transform !== nextTransform) {
        img.style.transform = nextTransform;
      }
    }
    updateCursor(img);

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
    zoomState.dragging = false;
    zoomState.suppressNextClick = false;
    applyTransform();
  }

  function installDelegatedListeners() {
    if (window._sam3MaskStudioDelegatedInstalled) return;
    window._sam3MaskStudioDelegatedInstalled = true;

    // Prevent native HTML5 image drag-and-drop inside preview so panning never gets hijacked
    document.addEventListener(
      "dragstart",
      function (e) {
        if (insidePreview(e.target)) {
          e.preventDefault();
          e.stopPropagation();
        }
      },
      true
    );

    document.addEventListener(
      "load",
      function (e) {
        const t = e.target;
        if (!t || t.tagName !== "IMG" || !insidePreview(t)) return;
        t.setAttribute("draggable", "false");
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

    // 2. Shift+Drag Box Select & Drag to Pan (Middle-button, Space + Left-button, or Left-button when zoomed in)
    const boxState = {
      active: false,
      startX: 0,
      startY: 0,
      curX: 0,
      curY: 0,
      overlayEl: null,
    };

    function ensureBoxOverlay() {
      if (boxState.overlayEl && document.body.contains(boxState.overlayEl)) {
        return boxState.overlayEl;
      }
      const el = document.createElement("div");
      el.id = "sam3_ms_box_overlay";
      el.style.position = "fixed";
      el.style.border = "2px dashed #00c8ff";
      el.style.background = "rgba(0, 200, 255, 0.14)";
      el.style.pointerEvents = "none";
      el.style.zIndex = "99999";
      el.style.display = "none";
      document.body.appendChild(el);
      boxState.overlayEl = el;
      return el;
    }

    function updateBoxOverlay(x0, y0, x1, y1) {
      const el = ensureBoxOverlay();
      const left = Math.min(x0, x1);
      const top = Math.min(y0, y1);
      const width = Math.abs(x1 - x0);
      const height = Math.abs(y1 - y0);
      el.style.left = left + "px";
      el.style.top = top + "px";
      el.style.width = width + "px";
      el.style.height = height + "px";
      el.style.display = width >= 2 && height >= 2 ? "block" : "none";
    }

    function hideBoxOverlay() {
      if (boxState.overlayEl) {
        boxState.overlayEl.style.display = "none";
      }
    }

    function getRenderedImageRect(img) {
      const rect = img.getBoundingClientRect();
      const natW = img.naturalWidth || 0;
      const natH = img.naturalHeight || 0;
      if (!natW || !natH || rect.width <= 0 || rect.height <= 0) {
        return rect;
      }
      const natRatio = natW / natH;
      const boxRatio = rect.width / rect.height;
      let renderedW = rect.width;
      let renderedH = rect.height;
      let offsetX = 0;
      let offsetY = 0;
      if (boxRatio > natRatio) {
        renderedH = rect.height;
        renderedW = rect.height * natRatio;
        offsetX = (rect.width - renderedW) * 0.5;
      } else {
        renderedW = rect.width;
        renderedH = rect.width / natRatio;
        offsetY = (rect.height - renderedH) * 0.5;
      }
      return {
        left: rect.left + offsetX,
        top: rect.top + offsetY,
        right: rect.left + offsetX + renderedW,
        bottom: rect.top + offsetY + renderedH,
        width: renderedW,
        height: renderedH,
      };
    }

    function submitBoxPayload(img, cX0, cY0, cX1, cY1) {
      if (!img) return;
      const rect = getRenderedImageRect(img);
      if (!rect || rect.width <= 0 || rect.height <= 0) return;
      const natW = img.naturalWidth || rect.width;
      const natH = img.naturalHeight || rect.height;
      const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
      const rx0 = clamp(((Math.min(cX0, cX1) - rect.left) / rect.width) * natW, 0, natW);
      const ry0 = clamp(((Math.min(cY0, cY1) - rect.top) / rect.height) * natH, 0, natH);
      const rx1 = clamp(((Math.max(cX0, cX1) - rect.left) / rect.width) * natW, 0, natW);
      const ry1 = clamp(((Math.max(cY0, cY1) - rect.top) / rect.height) * natH, 0, natH);
      if (rx1 - rx0 < 3 || ry1 - ry0 < 3) return;

      const root = getRoot();
      const ta = root ? root.querySelector("#sam3_ms_box_payload textarea, #sam3_ms_box_payload input") : null;
      if (!ta) return;
      ta.value = JSON.stringify({
        x0: Math.round(rx0),
        y0: Math.round(ry0),
        x1: Math.round(rx1),
        y1: Math.round(ry1),
        width: natW,
        height: natH,
        ts: Date.now(),
      });
      ta.dispatchEvent(new Event("input", { bubbles: true }));
      ta.dispatchEvent(new Event("change", { bubbles: true }));
    }

    const brushState = {
      active: false,
      erase: false,
      radius: 20,
      points: [],
      canvasEl: null,
      ctx: null,
    };

    function getBrushModeInfo() {
      const root = getRoot();
      if (!root) return { active: false, erase: false, radius: 20 };
      const modeContainer = root.querySelector("#sam3_ms_click_mode");
      const inputs = modeContainer ? modeContainer.querySelectorAll("input[type=radio]") : null;
      let idx = -1;
      if (inputs) {
        for (let i = 0; i < inputs.length; i++) {
          if (inputs[i].checked) {
            idx = i;
            break;
          }
        }
      }
      if (idx !== 2 && idx !== 3) {
        return { active: false, erase: false, radius: 20 };
      }
      const sizeInput = root.querySelector("#sam3_ms_brush_size input[type=number], #sam3_ms_brush_size input[type=range]");
      const radius = sizeInput ? Math.max(2, Number(sizeInput.value) || 20) : 20;
      return { active: true, erase: idx === 3, radius: radius };
    }

    function ensureBrushCanvas() {
      if (brushState.canvasEl && document.body.contains(brushState.canvasEl)) {
        return brushState.canvasEl;
      }
      const cvs = document.createElement("canvas");
      cvs.id = "sam3_ms_brush_canvas";
      cvs.style.position = "fixed";
      cvs.style.left = "0";
      cvs.style.top = "0";
      cvs.style.width = "100vw";
      cvs.style.height = "100vh";
      cvs.style.pointerEvents = "none";
      cvs.style.zIndex = "99998";
      cvs.style.display = "none";
      document.body.appendChild(cvs);
      brushState.canvasEl = cvs;
      brushState.ctx = cvs.getContext("2d");
      return cvs;
    }

    function startBrushOverlay(img, x, y, erase, radius) {
      const cvs = ensureBrushCanvas();
      cvs.width = window.innerWidth;
      cvs.height = window.innerHeight;
      cvs.style.display = "block";
      const rect = getRenderedImageRect(img);
      const natW = img.naturalWidth || rect.width || 1;
      const scale = rect.width / natW;
      const ctx = brushState.ctx;
      ctx.clearRect(0, 0, cvs.width, cvs.height);
      ctx.strokeStyle = erase ? "rgba(255, 70, 70, 0.7)" : "rgba(0, 210, 255, 0.7)";
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = Math.max(2, radius * 2 * scale);
      ctx.beginPath();
      ctx.arc(x, y, Math.max(1, radius * scale), 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(x, y);
    }

    function extendBrushOverlay(x, y) {
      if (!brushState.ctx) return;
      brushState.ctx.lineTo(x, y);
      brushState.ctx.stroke();
    }

    function finishBrushOverlay(img) {
      if (brushState.canvasEl) {
        brushState.canvasEl.style.display = "none";
      }
      if (!img || brushState.points.length === 0) return;
      const rect = getRenderedImageRect(img);
      if (!rect || rect.width <= 0 || rect.height <= 0) return;
      const natW = img.naturalWidth || rect.width;
      const natH = img.naturalHeight || rect.height;
      const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
      const pts = brushState.points.map((pt) => [
        Math.round(clamp(((pt[0] - rect.left) / rect.width) * natW, 0, natW - 1)),
        Math.round(clamp(((pt[1] - rect.top) / rect.height) * natH, 0, natH - 1)),
      ]);
      const root = getRoot();
      const ta = root ? root.querySelector("#sam3_ms_brush_payload textarea, #sam3_ms_brush_payload input") : null;
      if (!ta) return;
      ta.value = JSON.stringify({
        points: pts,
        radius: brushState.radius,
        erase: brushState.erase,
        width: natW,
        height: natH,
        ts: Date.now(),
      });
      ta.dispatchEvent(new Event("input", { bubbles: true }));
      ta.dispatchEvent(new Event("change", { bubbles: true }));
    }

    document.addEventListener(
      "mousedown",
      function (e) {
        if (!insidePreview(e.target)) return;
        const currentImg = getPreviewImg();
        if (!currentImg) return;

        if (e.shiftKey && e.button === 0 && !zoomState.spaceHeld) {
          boxState.active = true;
          boxState.startX = e.clientX;
          boxState.startY = e.clientY;
          boxState.curX = e.clientX;
          boxState.curY = e.clientY;
          zoomState.suppressNextClick = true;
          updateBoxOverlay(e.clientX, e.clientY, e.clientX, e.clientY);
          e.preventDefault();
          return;
        }

        const brushInfo = getBrushModeInfo();
        if (brushInfo.active && e.button === 0 && !zoomState.spaceHeld) {
          brushState.active = true;
          brushState.erase = brushInfo.erase;
          brushState.radius = brushInfo.radius;
          brushState.points = [[e.clientX, e.clientY]];
          zoomState.suppressNextClick = true;
          startBrushOverlay(currentImg, e.clientX, e.clientY, brushInfo.erase, brushInfo.radius);
          e.preventDefault();
          return;
        }

        zoomState.movedDist = 0;
        if (
          e.button === 1 ||
          (e.button === 0 && (zoomState.spaceHeld || zoomState.scale > 1.001))
        ) {
          zoomState.dragging = true;
          zoomState.startX = e.clientX;
          zoomState.startY = e.clientY;
          zoomState.startTx = zoomState.tx;
          zoomState.startTy = zoomState.ty;
          if (e.button === 1 || zoomState.spaceHeld) {
            zoomState.suppressNextClick = true;
            e.preventDefault();
          }
        }
      },
      true
    );

    window.addEventListener(
      "mousemove",
      function (e) {
        if (boxState.active) {
          boxState.curX = e.clientX;
          boxState.curY = e.clientY;
          updateBoxOverlay(boxState.startX, boxState.startY, boxState.curX, boxState.curY);
          return;
        }
        if (brushState.active) {
          brushState.points.push([e.clientX, e.clientY]);
          extendBrushOverlay(e.clientX, e.clientY);
          return;
        }
        if (!zoomState.dragging) return;
        const dx = e.clientX - zoomState.startX;
        const dy = e.clientY - zoomState.startY;
        const dist = Math.hypot(dx, dy);
        if (dist > zoomState.movedDist) {
          zoomState.movedDist = dist;
        }
        if (zoomState.movedDist >= PAN_THRESHOLD_PX) {
          zoomState.suppressNextClick = true;
          zoomState.tx = zoomState.startTx + dx;
          zoomState.ty = zoomState.startTy + dy;
          applyTransform();
        }
      },
      true
    );

    window.addEventListener(
      "mouseup",
      function (e) {
        if (boxState.active) {
          boxState.active = false;
          hideBoxOverlay();
          submitBoxPayload(getPreviewImg(), boxState.startX, boxState.startY, e.clientX, e.clientY);
          setTimeout(() => {
            zoomState.suppressNextClick = false;
          }, 250);
          return;
        }
        if (brushState.active) {
          brushState.active = false;
          finishBrushOverlay(getPreviewImg());
          brushState.points = [];
          setTimeout(() => {
            zoomState.suppressNextClick = false;
          }, 250);
          return;
        }
        if (!zoomState.dragging) return;
        const wasPanning =
          zoomState.movedDist >= PAN_THRESHOLD_PX || zoomState.spaceHeld;
        zoomState.dragging = false;
        applyTransform();

        if (wasPanning) {
          zoomState.suppressNextClick = true;
          setTimeout(() => {
            zoomState.suppressNextClick = false;
          }, 250);
        }
      },
      true
    );

    // Strictly block Gradio .select() click when finishing a pan drag or holding Space
    document.addEventListener(
      "click",
      function (e) {
        if (!insidePreview(e.target)) return;
        if (
          (zoomState.suppressNextClick ||
            zoomState.spaceHeld ||
            zoomState.movedDist >= PAN_THRESHOLD_PX) &&
          !e._sam3SyntheticRightClick
        ) {
          zoomState.suppressNextClick = false;
          zoomState.movedDist = 0;
          e.stopPropagation();
          e.stopImmediatePropagation();
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

      if (zoomState.movedDist >= PAN_THRESHOLD_PX) {
        zoomState.movedDist = 0;
        return;
      }

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

    // 4. Keyboard shortcuts: Ctrl+Z -> Undo, Space -> Safe Pan Mode
    document.addEventListener("keydown", function (e) {
      if (isTypingTarget(e.target)) return;
      const container = getPreviewContainer();
      if (!container || container.offsetParent === null) return;

      if (e.code === "Space" && !zoomState.spaceHeld) {
        zoomState.spaceHeld = true;
        updateCursor(getPreviewImg());
      }

      if (
        (e.ctrlKey || e.metaKey) &&
        !e.shiftKey &&
        !e.altKey &&
        (e.key === "z" || e.key === "Z")
      ) {
        const root = getRoot();
        const undoBtn = root ? root.querySelector("#sam3_ms_undo_btn") : null;
        if (undoBtn) {
          e.preventDefault();
          e.stopPropagation();
          undoBtn.click();
        }
        return;
      }

      if (e.ctrlKey || e.metaKey || e.altKey) return;

      const key = (e.key || "").toLowerCase();
      const root = getRoot();
      if (!root) return;

      if (key === "i") {
        const invertBtn = root.querySelector("#sam3_ms_invert_btn");
        if (invertBtn) {
          e.preventDefault();
          invertBtn.click();
        }
      } else if (key === "r") {
        e.preventDefault();
        resetZoom();
      } else if (key === "1" || key === "2" || key === "3" || key === "4") {
        const modeContainer = root.querySelector("#sam3_ms_click_mode");
        const inputs = modeContainer ? modeContainer.querySelectorAll("input[type=radio]") : null;
        const idx = Number(key) - 1;
        if (inputs && inputs[idx]) {
          e.preventDefault();
          inputs[idx].click();
        }
      } else if (key === "v") {
        const viewContainer = root.querySelector("#sam3_ms_preview_mode");
        const inputs = viewContainer ? viewContainer.querySelectorAll("input[type=radio]") : null;
        if (inputs && inputs.length > 0) {
          e.preventDefault();
          let activeIdx = 0;
          for (let i = 0; i < inputs.length; i++) {
            if (inputs[i].checked) {
              activeIdx = i;
              break;
            }
          }
          inputs[(activeIdx + 1) % inputs.length].click();
        }
      }
    });

    document.addEventListener("keyup", function (e) {
      if (e.code === "Space" && zoomState.spaceHeld) {
        zoomState.spaceHeld = false;
        updateCursor(getPreviewImg());
      }
    });

    window.addEventListener("blur", function () {
      if (zoomState.spaceHeld || zoomState.dragging) {
        zoomState.spaceHeld = false;
        zoomState.dragging = false;
        updateCursor(getPreviewImg());
      }
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
