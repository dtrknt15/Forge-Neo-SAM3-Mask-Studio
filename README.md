# Forge Neo - SAM3 Mask Studio

Standalone **SAM3 (Segment Anything Model 3)** mask creation, interactive refinement, and export studio for **Stable Diffusion WebUI Forge - Neo**.

[English](#english) | [日本語](#japanese)

---

<a id="english"></a>
## English

### Overview
**SAM3 Mask Studio** adds a dedicated **`SAM3 Mask`** tab to Forge Neo that connects SAM3's open-vocabulary text segmentation and interactive click segmentation directly to your `img2img -> Inpaint upload` workflow and RGBA cutout export.

### Key Features
- **Open-Vocabulary Text Masking & Presets**:
  - Segment objects with natural language positive/negative prompts (`person`, `hair`, `face`, or descriptive phrases).
  - Built-in presets (`full_body`, `face`, `hair`, `background`) and custom preset management saved to `user_presets.json` (so `git pull` updates never conflict with your custom presets).
  - Remove individual applied prompts at any time by unchecking their prompt chips.
- **Interactive Click Add / Exclude + Zoom & Pan**:
  - **Left-Click** on the preview image: Add region (or follow selected Click Mode).
  - **Right-Click** on the preview image: Immediately **Exclude (subtract)** region without leaving your cursor position.
  - **Mouse Wheel**: Zoom in/out (`100%` – `800%`) centered at the cursor; zoom & pan position persist across clicks so you can refine tiny details effortlessly.
  - **Drag**: Pan the zoomed preview image (or middle-mouse drag at any zoom); **Double-Click** or click **Reset Zoom** to return to `100%`.
- **Unified Set-Operation Engine**:
  $$\text{final\_mask} = (\text{base\_mask} \cup \text{positive\_prompts} \cup \text{positive\_clicks}) \setminus (\text{negative\_prompts} \cup \text{negative\_clicks})$$
- **Mask Refinement**:
  - **Invert Mask**: One-click full mask inversion (`0 <-> 255`); subsequent clicks/prompts naturally add to or subtract from the inverted mask.
  - **Grow / Shrink (px)**: Elliptical morphological dilation/erosion (`-64` to `+64` px).
  - **Detection Score Threshold**: Fine-tune SAM3 text detection sensitivity (`0.05` – `0.95`).
  - **Multi-Step Undo**: Full state snapshot history across prompts, clicks, inversion, and clear.
- **One-Click Import & Output**:
  - **Load Image from img2img**: Pull the current image from `img2img` directly into SAM3 Mask Studio.
  - **Send to Inpaint upload**: Transfers both the original image and the generated mask into `img2img -> Inpaint upload` and switches directly to the `Inpaint upload` sub-tab.
  - **Download Masked PNG**: Exports an RGBA PNG (`Full Image` or `Crop to Mask`, with `Transparent` or `White Background`).
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
- **自然言語テキストマスク ＆ プリセット機能**:
  - ポジティブ／ネガティブプロンプトによる領域抽出と集合演算合成（`positive_set - negative_set`）。
  - デフォルトプリセット（人物全体・顔・髪・背景）に加え、UI からカスタムプリセットを保存可能（Git 管理外の `user_presets.json` に保存されるため、`git pull` による拡張アップデート時に競合しません）。
  - 適用中プロンプトはチップ一覧のチェック解除でいつでも個別に削除・再計算できます。
- **直感的なプレビュー操作（右クリック除外・ホイール拡大・ドラッグ移動）**:
  - **左クリック**: クリックした領域をマスクに追加（または選択中のクリックモードに従う）
  - **右クリック**: モードを切り替えずに、その場の対象を即座に **除外（くり抜き）**
  - **マウスホイール**: カーソル位置を中心に **100%〜800% 拡大・縮小**（拡大したまま連続クリックしてもズーム位置が維持されます）
  - **ドラッグ移動**: 拡大中のドラッグ（または中ボタンドラッグ）で画面移動。**ダブルクリック** または **「拡大リセット」** で 100% 表示に復帰します。
- **マスク調整・反転・アンドゥ**:
  - **マスクを反転**: 現在のマスク範囲をワンクリックで反転（反転した状態に対して、さらにクリック追加／除外を重ねられます）。
  - **拡張 / 縮小 (px)**: `-64px 〜 +64px` の楕円カーネルによる境界の拡張・収縮。
  - **検出スコア閾値**: テキストプロンプトの検出感度を `0.05 〜 0.95` で調整可能。
  - **元に戻す (Undo)**: プロンプト・クリック・反転・クリアを含む操作履歴（直近10件）をスナップショット復元。
- **ワンクリック入出力**:
  - **img2img から画像を取得**: `img2img` 上の画像をワンクリックで読み込み。
  - **Inpaint upload へ転送**: 元画像とマスクを `img2img` の `Inpaint upload` へ同時セットし、直接 `Inpaint upload` タブへジャンプします。
  - **マスク適用PNGをDL**: 全体またはマスク領域でクロップした透過 PNG（または白背景 PNG）を書き出し・ダウンロード。

### 導入方法
1. Forge Neo の **Extensions** -> **Install from URL** を開きます。
2. `https://github.com/dtrknt15/Forge-Neo-SAM3-Mask-Studio.git` を入力して **Install** を押し、**Installed** タブから **Apply and restart UI** を実行します。
3. [Hugging Face (facebook/sam3)](https://huggingface.co/facebook/sam3) でライセンスを承認し、**SAM3 マスク** タブ内の「モデルセットアップ」からダウンロードするか、`models/sam3/`（または `models/SAM3/`）に `sam3.pt` を配置してください。

### ライセンス
- 本拡張機能のソースコードは [MIT License](./LICENSE) の下で提供されます。
- SAM3 のモデル重みは同梱しておらず、Meta の [SAM License](https://huggingface.co/facebook/sam3) に従います。
