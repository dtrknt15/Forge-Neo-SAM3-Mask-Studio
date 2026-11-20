# Forge Neo - SAM3 Mask Studio

Standalone **SAM3 (Segment Anything Model 3)** mask creation, interactive refinement, and export studio for **Stable Diffusion WebUI Forge - Neo**.

[English](#english) | [日本語](#japanese)

---

<a id="english"></a>
## English

### Overview
**SAM3 Mask Studio** adds a dedicated **`SAM3 Mask`** tab to Forge Neo that connects SAM3's open-vocabulary text segmentation and interactive click segmentation directly to your `img2img -> Inpaint upload` workflow and RGBA cutout export.

### Key Features
- **Open-Vocabulary Text Masking, Instance Picker & Presets**:
  - Segment objects with natural language positive/negative prompts (`person`, `hair`, `face`, or descriptive phrases).
  - **Detected Instances (Instance Picker)**: Select or exclude individual detected instances (with confidence scores) per positive prompt.
  - Built-in presets (`full_body`, `face`, `hair`, `background`) and custom preset management saved to `user_presets.json` (so `git pull` updates never conflict with your custom presets).
  - Remove individual applied prompts at any time by unchecking their prompt chips.
- **Interactive Click, Multi-Point Refine, Box, Soft Brush & Zoom/Pan**:
  - **Left-Click**: Add or exclude region according to the selected mode.
  - **Right-Click**: Immediately **Exclude (subtract)** region without switching modes.
  - **Multi-point Object Refine**: Toggle to refine a single object iteratively using multiple positive/negative points, then click **Commit Object** (or **Undo Point** / **Clear Points**).
  - **Shift + Drag**: Draw a **bounding box** to segment or exclude an object in one drag.
  - **Brush: Add / Erase**: Freehand brush & eraser with adjustable **Brush Size (px)** (`2` – `120` px) and **Brush Hardness** (`0.10` – `1.00`) for soft-edge touch-ups.
  - **Mouse Wheel**: Zoom in/out (`100%` – `800%`) centered at the cursor; zoom & pan position persist across edits.
  - **Drag / Space + Drag**: Pan the zoomed preview (`Space + Drag` or `Middle-Drag`); **Double-Click** or press `R` / **Reset Zoom** to return to `100%`.
  - **Preview View & Opacity**: Switch between `Overlay`, `Mask Only`, `Cutout` (checkerboard), `Outline`, and `Original`, and adjust **Overlay Opacity** (`0.10` – `0.90`).
  - **Keyboard Shortcuts**: `Ctrl+Z` (Undo), `I` (Invert Mask), `R` (Reset Zoom), `V` (Cycle Preview View), `1`–`4` (Switch Click/Brush Mode).
- **Unified Set-Operation Engine**:
  - `final_mask = (base_mask ∪ positive_prompts ∪ positive_clicks) \ (negative_prompts ∪ negative_clicks)`
- **Mask Refinement**:
  - **Invert Mask**: One-click full mask inversion (`0 <-> 255`); subsequent clicks/prompts naturally add to or subtract from the inverted mask.
  - **Grow / Shrink (px)**: Elliptical morphological dilation/erosion (`-64` to `+64` px).
  - **Feather / Blur (px)**: Gaussian edge feathering (`0` – `64` px) reflected in preview, `Inpaint upload`, and PNG export.
  - **Edge Refine (Guided Radius)** & **Edge Smooth (px)**: Fast edge-aware Guided Filter refinement (`0` – `32`) and contour smoothing (`0` – `16` px).
  - **Fill Holes & Remove Small Islands**: One-click interior hole filling and connected-component noise removal (`Min Island Area`).
  - **Detection Score Threshold**: Fine-tune SAM3 text detection sensitivity (`0.05` – `0.95`).
  - **Multi-Step Undo (`Ctrl + Z`)**: zlib-compressed state snapshot history across prompts, clicks, boxes, brush strokes, adjustments, inversion, and clear.
- **Import, Export, img2img Auto-Mask & Batch Processing**:
  - **Send to SAM3 Mask (`🎭`)** & **Load Image from img2img** & **Import Mask PNG**: Send generated images directly from `txt2img` / `img2img` / `extras` galleries, pull the active `img2img` image, or load an existing mask PNG as a base mask for further SAM3 refinement.
  - **Send to Inpaint upload**: Transfers both the original image and the generated mask (including feathering/guided refine) into `img2img -> Inpaint upload` and switches directly to the `Inpaint upload` sub-tab (with optional **Auto-unload SAM3 model** to free VRAM, plus **VRAM Status** inspection).
  - **Download Masked PNG** & **Export Layers (ZIP)**: Exports `Full Image` or `Crop to Mask` PNGs (`Transparent`, `White Background`, `Mask Only (B&W)`), or packages the final mask, RGBA cutout, and per-prompt mask layers into a `.zip` archive.
  - **SAM3 Auto-Mask (`img2img` Script)**: Built-in `AlwaysVisible` accordion in `img2img` to auto-generate and apply inpaint masks from text prompts during generation.
  - **Batch Processing**: Apply text prompts or presets across an entire folder with **Cancel Batch**, **Recursive (Subfolders)**, **Skip Existing (Resume)**, **Auto-unload SAM3 after batch**, **Save JSON Metadata**, Grow/Feather/Edge Refine/Smooth/Threshold, Fill Holes, Remove Small Islands, Invert Mask, and Skip Empty Masks.
- **BYOT (Bring Your Own Token) Setup**:
  - Safe Hugging Face access check and user-triggered download for gated `facebook/sam3` weights, plus manual placement support in `<Forge-Neo>/models/sam3/`.

### Requirements
- **Stable Diffusion WebUI Forge - Neo** (Python 3.12+, PyTorch 2.7+, CUDA GPU)
- **Windows Note**: SAM3 requires `triton` (e.g. `triton-windows` bundled in Forge Neo environments).

### Installation
1. In Forge Neo, open **Extensions** -> **Install from URL**.
2. Paste `https://github.com/dtrknt15/Forge-Neo-SAM3-Mask-Studio.git` and click **Install**.
3. Go to **Extensions** -> **Installed** and click **Apply and restart UI**.

### Model Setup (BYOT)
1. Visit [facebook/sam3 on Hugging Face](https://huggingface.co/facebook/sam3) and accept the SAM License.
2. Open the **SAM3 Mask** tab -> **Model Setup**, enter your Hugging Face token (or leave blank to use your locally logged-in token), click **Check Access**, and click **Download SAM3 Model**.
3. Or place `sam3.pt` manually into `<Forge-Neo>/models/sam3/`.

### Adding Translations
This extension uses WebUI's standard `localizations/*.json` mechanism:
1. Copy `localizations/ja_JP.json` to `localizations/<locale>.json` (e.g., `localizations/zh_CN.json`, `localizations/ko_KR.json`).
2. Translate the JSON values while keeping all keys and `{...}` placeholders unchanged.
3. Open a Pull Request using `.github/PULL_REQUEST_TEMPLATE/translation.md`.

---

<a id="japanese"></a>
## 日本語

### 概要
**SAM3 Mask Studio** は、Forge Neo 上に独立タブ **「SAM3 Mask（SAM3 マスク）」** を追加し、SAM3（Segment Anything Model 3）による自然言語テキストマスク・クリック追加／除外・マスク調整・`Inpaint upload` への直接転送・透過切り抜き PNG 書き出しまでを一貫して行える拡張機能です。

### 主な機能
- **自然言語テキストマスク・インスタンス個別選択・プリセット機能**:
  - ポジティブ／ネガティブプロンプトによる領域抽出と集合演算合成（`positive_set - negative_set`）。
  - **検出インスタンス個別選択 (`Instance Picker`)**: 検出された複数オブジェクトをスコア付きチェックボックスで個別に有効／除外できます。
  - デフォルトプリセット（人物全体・顔・髪・背景）に加え、UI からカスタムプリセットを保存可能（Git 管理外の `user_presets.json` に保存されるため、`git pull` による拡張アップデート時に競合しません）。
  - 適用中プロンプトはチップ一覧のチェック解除でいつでも個別に削除・再計算できます。
- **直感的なプレビュー操作（クリック・複数点リファイン・矩形選択・ソフトブラシ・拡大移動）**:
  - **左クリック / 右クリック**: 選択中モードで領域追加／除外（右クリックはモード切替なしで即 **除外**）
  - **複数点リファインモード (`Multi-point Object Refine`)**: 1つの対象に対して追加点・除外点を打ち足して輪郭を追い込み、`対象を確定`（または `直前の点を戻す` / `点をクリア`）できます。
  - **矩形選択（`Shift + ドラッグ`）**: ドラッグした矩形範囲を SAM3 で一括追加／除外
  - **ブラシ / 消しゴム**: `Brush Size (px)`（`2〜120 px`）と `ブラシ硬さ`（`0.10〜1.00`）調整対応のソフトブラシ＆フリーハンド微調整
  - **マウスホイール / ドラッグ移動**: カーソル中心 **100%〜800% 拡大・縮小**、拡大中ドラッグ・`Space + ドラッグ`・中ボタンドラッグで移動、**ダブルクリック** または `R` / **拡大リセット** で 100% 復帰
  - **プレビュー表示切替 ＆ 不透明度**: `オーバーレイ / マスク単体 / 切り抜き (市松模様) / 輪郭 / 元画像` の比較切替と不透明度調整（`0.10〜0.90`）
  - **ショートカット**: `Ctrl+Z`（元に戻す）、`I`（反転）、`R`（ズームリセット）、`V`（表示モード切替）、`1`〜`4`（クリック/ブラシモード切替）
- **マスク調整・反転・アンドゥ**:
  - **マスクを反転**: 現在のマスク範囲をワンクリックで反転（反転後さらにクリック・ブラシ重ね掛け可能）
  - **拡張 / 縮小 (px)**: `-64px 〜 +64px` の楕円カーネルによる境界の拡張・収縮
  - **ぼかし / フェザー (px)**: `0px 〜 64px` のガウシアンフェザリング（プレビュー・`Inpaint upload` 転送・PNG出力に反映）
  - **エッジ精細化 (Guided半径)** ＆ **輪郭スムージング (px)**: 元画像のエッジに沿わせる高速 Guided Filter（`0〜32`）と輪郭平滑化（`0〜16 px`）
  - **穴埋め ＆ 小ノイズ除去**: マスク内の穴埋めと指定面積未満の孤立ノイズ一括除去
  - **検出スコア閾値**: テキストプロンプトの検出感度を `0.05 〜 0.95` で調整可能
  - **元に戻す (`Ctrl + Z`)**: zlib 圧縮による省メモリな操作履歴（直近10件）をスナップショット復元
- **入出力・img2img 自動マスク ＆ バッチ処理**:
  - **ギャラリーから送信 (`🎭`)**・**img2img から画像を取得**・**既存マスクPNG読込**: `txt2img` / `img2img` / `extras` ギャラリーからのワンクリック送信、`img2img` からの画像取得、既存マスクの読み込み・再編集に対応
  - **Inpaint upload へ転送**: 元画像とマスク（フェザー・Guided精細化反映済み）を `img2img` の `Inpaint upload` へ同時セットし、直接タブへジャンプ（転送後の **SAM3 モデル自動アンロード / VRAM 解放** オプション・`VRAM 状態` 表示付き）
  - **マスク適用PNGをDL** ＆ **レイヤー一括出力 (ZIP)**: 全体／マスク領域クロップで `透過`・`白背景`・`マスクのみ (白黒)` PNG を書き出し、または最終マスク・RGBA切り抜き・プロンプト別レイヤーを ZIP でまとめて出力
  - **SAM3 Auto-Mask (`img2img` Script)**: `img2img` 画面の `SAM3 Auto-Mask` アコーディオンから、生成時にテキストプロンプトで自動マスク生成・適用が可能
  - **バッチ処理（フォルダ一括）**: `バッチ中断`、`サブフォルダを含める`、`既存ファイルをスキップ (再開)`、`バッチ完了後に SAM3 を自動アンロード`、`JSON メタデータを保存`、各種調整（拡張/ぼかし/Guided精細化/輪郭スムージング/閾値/穴埋め/小ノイズ除去/反転/空マスクスキップ）に対応

### 導入方法
1. Forge Neo の **Extensions** -> **Install from URL** を開きます。
2. `https://github.com/dtrknt15/Forge-Neo-SAM3-Mask-Studio.git` を入力して **Install** を押し、**Installed** タブから **Apply and restart UI** を実行します。
3. [Hugging Face (facebook/sam3)](https://huggingface.co/facebook/sam3) でライセンスを承認し、**SAM3 マスク** タブ内の「モデルセットアップ」からダウンロードするか、`models/sam3/`（または `models/SAM3/`）に `sam3.pt` を配置してください。

### ライセンス
- 本拡張機能のソースコードは [MIT License](./LICENSE) の下で提供されます。
- SAM3 のモデル重みは同梱しておらず、Meta の [SAM License](https://huggingface.co/facebook/sam3) に従います。
