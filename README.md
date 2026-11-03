# ComfyUI H3 Native Production

用 ComfyUI 操作原生 LightX2V / MiniMax H3：單段完整生成、學習式潛空間 2× 放大，以及 V2 兩段接續。這是針對既有雙卡生產環境整理的部署套件，並非 MiniMax 或 LightX2V 官方插件。

**本套件不包含模型、影片、參考圖片、憑證或使用者資料。** 需要先備妥 Linux/CUDA、相容原生 H3 Python 環境及指定混合精度權重；不是任意 ComfyUI 環境的一鍵安裝包。不要把整個儲存庫直接放入 `custom_nodes`。

## 包含什麼

| 入口 | 用途 |
| --- | --- |
| `H3-01-單段完整流程-雙卡10+5.json` | 單段 5–15 秒；一採後自動放大及二採，只交付最後成片 |
| `H3-02-V2統一創作-自動分段10+5.json` | 最多兩段；結構化創作腳本分配素材、提示詞、故事版，再接續及合併 |
| `plugins/ComfyUI-H3-FeiHou-Native` | 原圖載入、完整生成及內部接續邏輯 |
| `plugins/ComfyUI-H3-Unified-Creation` | V2 腳本編譯器、素材滑塊和生成節點 |
| `plugins/ComfyUI-H3-Native-Continuation` | 相容的共享後端模組；不註冊已退役的獨立接續節點 |
| `plugins/ComfyUI-LightX2V-RemoteH3` | 保留所需參考素材收集節點；此發行版未包含舊遠端 HTTP 提交器 |
| `runtime/`、`support/` | 配套原生推理程式、混合精度適配、交接及學習式放大程式 |

節點封裝了完整流程，工作流畫布不會展開成採樣器、VAE 等數十個節點。V2 讀取明確的 JSON 腳本，不會用語言模型猜測任意提示詞如何分段。範例在 `plugins/ComfyUI-H3-Unified-Creation/examples/`。

## 固定採樣配置

- 兩張 GPU，原生 TP2；BF16 文本編碼器，FeiHou INT8/BF16 混合主模型。
- 一採：原生完整 10 步，不載入外部加速 LoRA。
- 交接：一採輸出的 clean latent 經指定學習式模型在空間上放大 2×。
- 二採：原始 **8 步時間表**，從索引 **3** 開始，實際執行 **5 次評估**。不是重新建立的 5 步網格，也不是官方完整 8 步生成。
- 二採使用原生 `minimax_h3_ref2v_turbo_8step_v1.1_768p_bf16.safetensors`，`training_euler`、video/audio shift 12/3、LoRA strength 1.0、alpha 8、CFG 關閉。
- `NoCaching`、compile 關閉；保留記憶體、取消、鎖定、雜湊和輸入綁定保護。

成片寬高須為 64 倍數，長邊 1080–1920、短邊不超過 1280、總像素不超過 1920×1088。一採寬高各為成片一半。常用尺寸如 1920×1088、1088×1920、1280×768、768×1280；這些是對齊尺寸，比例近似，程式不另行裁剪或補幀。

時長保留模型原生幀數。名義 30 秒使用兩段及 22 幀重疊，目前實際為 702 幀 / 24 fps，即 **29.25 秒**，不承諾恰好 30.000 秒。

## 安裝前準備

此版本從既有部署抽出，基準環境為 Linux、Python 3.12、PyTorch 2.11 CUDA 13、ComfyUI 0.43 / frontend 1.57；其他組合尚未做 GPU 相容性認證。ComfyUI 環境需提供 `torch`、`numpy`、`Pillow` 及 `comfy_api.latest`。原生環境另需 LightX2V 依賴、SageAttention2、Triton、ConvRot / int8 算子等，見 `runtime/source/pyproject.toml` 及原生安裝文檔。安裝工具不更動 Python 套件。

準備下列資產：

1. 完整 H3 Ref2AV 模型目錄，包括文本編碼器與音視頻 VAE。
2. 已對齊的 FeiHou 混合權重及 `alignment-manifest.json`。適配器要求 550 個主區塊權重，僅 fc2 層 29、39、44、45、49 為 BF16，其餘使用既定 INT8 ConvRot；任意量化模型不能直接代替。
3. 指定原生 8-step v1.1 768p BF16 LoRA，不能用 `_comfyui_` 版本。來源：[LightX2V Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/30f485ed874aa44c8183348ad361c5eabad318da)。
4. `h3_clean_latent_upscaler_film_epoch200.safetensors`；本包僅含其架構，需另外取得權重。
5. 可用的原生 Python venv（含 `bin/torchrun`），以及 `/usr/bin/ffmpeg`、`/usr/bin/ffprobe`。
6. 與模型、排程和 LoRA 完全綁定的 AdaLN caches。改變檢查點或路徑後應重建，不能搬來不匹配的 cache。

安裝工具校驗兩個指定資產的 SHA256：

```text
LoRA:     46c64fc94509489f18dd1f545172e30c8df7d4cc50a30e96744409b4288c937d
Upscaler: 984afb58f11d01274b90d880596ce2f93bc9c512db66fdeecd4e2d99b371d3e4
```

## 安裝

在伺服器上執行；倉庫與部署目錄須分開：

```bash
git clone https://github.com/kuangrenaigc-stack/ComfyUI-H3-Native-Production.git
cd ComfyUI-H3-Native-Production
cp settings.example.json settings.local.json
# 編輯 settings.local.json，填入本機模型、venv、ComfyUI 及輸出目錄。
python3 tools/install.py --config settings.local.json --dry-run
python3 tools/install.py --config settings.local.json
```

`runtime_root`、`support_root` 和四個插件目錄必須不存在；安裝器拒絕覆蓋現有部署。它複製程式、替換路徑，並對現有模型及 venv 建立連結；不下載權重、不啟停服務。`settings.local.json` 已被 Git 忽略。

準備 AdaLN cache 的工具在 `runtime/source/tools/cache_minimax_h3_adaln/`。分別使用 `runtime/production/two_pass.py` 的 `phase_config(config, 'draft')` 與 `phase_config(config, 'final')` 產生 10 步無 LoRA、8 步有 LoRA 的配置，再用相同模型及 CUDA 環境生成 cache。不要編輯 cache binding 來繞過驗證。尚未備妥這些原生依賴和 cache 時，安裝完成不代表可以推理。

將真正的成片目錄連到 **ComfyUI 實際 output 目錄** 下的 `H3-成片`，以便節點預覽讀取。例如 output 設為 `/srv/comfyui/output`：

```bash
ln -s /srv/h3/videos /srv/comfyui/output/H3-成片
```

若同名路徑已存在，先檢查它的用途，不要強制覆蓋。自行重啟 ComfyUI，從本包 `workflows/` 匯入兩份 JSON。安裝套件不會覆蓋已儲存的工作流。

## 使用

- **單段**：用「H3 原圖載入」上傳原圖，接參考素材節點，填完整六段提示詞，設定最後成片尺寸與時長後執行。圖片來源檔案及 SHA256 會保留，推理使用原檔；一般 IMAGE 節點無法代替此來源證明。
- **V2**：素材節點可調圖片 / 視頻 / 音頻數量；JSON 的 references 定義全局 slot，segments/shots 定義每段實際引用。編譯器再轉成本段局部編號，未引用素材不送入該段。單段最多 9 張圖、3 段視頻、3 段音頻，合計最多 12；全局上限 18/6/6，合計最多 24。
- 故事版作為圖片素材，透過腳本綁定對應分段；沒有獨立故事版上傳器。V2 視頻參考移除音軌，需要聲音時另接音頻素材。
- 範例 JSON 中的圖片檔名是佔位符，必須先上傳自己的圖片再執行。
- 成片按 `影片_NNNN.mp4` 編號，保留原生尺寸 / 時長。伺服器共享、Mac 桌面映射屬部署設定，本包不包含私人掛載資料。

## 檢查與限制

```bash
python3 tools/check_package.py
python3 -m unittest discover -s tools/tests
# 完整 V2 測試需在 ComfyUI Python 環境，且 ComfyUI 根目錄位於 PYTHONPATH：
python -m unittest discover -s plugins/ComfyUI-H3-Unified-Creation/tests
```

這次公開整理通過 76 項 V2 CPU 測試及 4 項安裝器測試，並驗證程式語法、JSON、工作流連線和私人資料掃描；**未對重新定位後的公開版做新的完整 GPU 出片測試**。既有部署不會由本套件自動升級。內含部分歷史診斷測試，只能在對應原生環境使用；不要直接執行 GPU smoke 腳本驗證安裝。

既有 MP4 的重新編碼放大不是此版本的成品入口；不要把兩段生成接續解讀為任意影片都可還原原始 latent。

## FFN 顯存優化與診斷

二採 FFN 預設按 16,384 個 token 分塊，使用原生混合精度算子與 LoRA；局部結果組合後仍只執行一次 TP 歸約。工具不會在日常生成時開啟 profiler。可選的官方 benchmark / 雙卡 trace 入口和鎖定方式見 [diagnostics/README](support/diagnostics/README.md)。

實測範圍與限制見 [FFN 驗證報告](docs/ffn-optimization.md)：大尺寸 FFN 診斷峰值約 14.27→8.08 GiB；完整既有交接二採通過且輸出逐位元組相同，但整體最高顯存沒有降低。

## English

Deployment bundle for an existing two-GPU native LightX2V H3 pipeline: native 10-step draft → learned clean-latent 2× → five evaluations on the original eight-step Turbo schedule, plus two-segment creation. Clone outside `custom_nodes`, supply the exact mixed-precision model assets and a compatible CUDA environment, then use `tools/install.py`. No weights or personal media are distributed. The relocated public bundle has CPU/static validation, not a fresh end-to-end GPU certification.

## 授權與來源

Apache-2.0 適用於本專案新寫的程式；第三方程式保留原授權，詳見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型權重不隨程式授權重新授權。
