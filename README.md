# data-redactor

data-redactor は GiNZA ベースの**日本語ドキュメント機密マスキング**ツールです。

LLM へ渡す前の前処理として、人名・社名・商標・連絡先（メール等）などの機密情報を検出し、
伏せ字（`[社1]` など）に置き換えることを目標にしています。

検出は次の合議で行います。

- マスク辞書（社名・商標・人名の名簿）
- Sudachi の品詞（人名・地名・固有名詞）
- GiNZA NER（`ja_ginza_electra` と `ja_ginza` の 2 モデル）
- 連絡先の正規表現（メール）

そのうえで確信度に応じて「自動マスク／要レビュー／非表示」に振り分けます。
**recall 最優先**（マスク漏れ＝漏洩）の設計で、不確実なものは伏せずに人のレビューへ回します。

入力は、`.txt` / `.md` / `.pdf` / `.docx` / `.xlsx` / `.pptx` などのファイル、貼り付けテキスト、
kb-mcp の文書に対応します（kb-mcp から移植した `DocumentLoader` でテキスト化 → チャンク分割 → 解析）。

> 設計・経緯の詳細はローカルの `docs-dev/`（git 管理外）と [CLAUDE.md](CLAUDE.md) を参照してください。

---

## セットアップ

```powershell
uv sync
```

Python 3.11 / spaCy 3.7 系 / numpy 1.x に固定しています
（`ja_ginza_electra` の依存と GiNZA 5.2 の制約のため。3.12 や spaCy 3.8・numpy 2 では動作しません）。

---

## Web UI（Streamlit）

```powershell
uv run data-redactor ui
```

ブラウザで http://localhost:8501 が開きます。上部のモードで画面を切り替えます。

- **🔒 マスキング** … 本ツールの主機能。
- **📒 マスク辞書** … 確定マスクする社名・商標・社員名の名簿を編集。
- **🚫 除外リスト** … マスク「しない」語の名簿を編集。
- **🗂 キャッシュ** … 解析（NER）をキャッシュ済みの文書を一覧・削除。

固有表現抽出（NER）の素の結果を見たいときは、サイドバーの **「🔍 NER ビューア（参考）」** トグルで開けます
（参考ツール。OFF でマスキングに戻ります）。

### マスキングの流れ

1. 入力方法（✏️ テキスト / 📄 ファイル / 📚 kb-mcp / 🗂 キャッシュから選択）を選ぶ。
2. **[🔍 解析する]** を押すと候補を検出します（GiNZA 2 モデルを使うので時間がかかります。
   進捗はステージ表示、所要時間は結果の先頭に出ます）。
3. 候補表で確信度フィルタをかけつつ、チェックでマスク対象を選び、**[✅ マスクを反映]** を押します。
   不要な誤検出は「除外」にチェック → **[🚫 選択を除外リストへ]** で以後どの文書でも候補外にできます。
4. 結果は「色付き／マスク済み／原文」で確認・ダウンロードできます。

確信度フィルタの既定は 確定・強・中・弱（**微弱・除外は既定で非表示**）。
表（`|` 区切り）は検出のときだけ平文化し、**マスクは `|` 入りの原文に当てて体裁を保持**します。

### kb-mcp 連携

「📚 kb-mcp から選択」を使うには、kb-mcp を HTTP サーバとして起動しておきます。

```powershell
# kb-mcp プロジェクト側で
uv run kb-mcp-server --transport http --port 8000
```

UI で URL（既定 `http://localhost:8000/mcp`）を指定し、「文書リストを取得」→ 文書を選択 →「解析する」。
本文は**チャンク単位**で取得します（kb-mcp は格納時に分割済みなので、結合せずそのまま解析します）。

---

## CLI

統一コマンド `data-redactor`（実体は [src/cli.py](src/cli.py)。`uv run main.py <サブコマンド>` でも可）。

```powershell
# Streamlit UI を起動
uv run data-redactor ui

# マスキング（ファイル or --text）。既定で data/mask_dict.yaml を自動読込
uv run data-redactor mask report.pdf
uv run data-redactor mask --text "本文をここに貼り付け"
uv run data-redactor mask report.docx --out masked.txt   # マスク済みを書き出し
uv run data-redactor mask report.docx --audit            # 候補の票分布・確信度（表層なし＝共有OK）
uv run data-redactor mask report.docx --audit-surface    # 監査に表層も付ける（機密・共有禁止）
uv run data-redactor mask report.docx --no-flatten       # 表の平文化を切る

# NER → displaCy の HTML（ner.html）。--open で既定ブラウザ表示・--serve でサーバ表示
uv run data-redactor ner report.pdf --open

# 各トークンの Sudachi 品詞 / NER ラベルを並べて観察（recall の穴を見る）
uv run data-redactor debug report.pdf --both-models --all-tokens

# 品質ゲート（ruff + mypy）
uv run data-redactor check
```

---

## マスク辞書・除外リスト・キャッシュ（ローカル専用）

機密のため、`data/*.yaml` と `data/cache.db` は **git 管理外**です。各マシンで用意します。

- **マスク辞書** `data/mask_dict.yaml`
  社名・商標・社員名の名簿。一致語は**確定マスク**（文書内の全出現）。
  別表記（英語↔カタカナ・略称）を 1 つの代表表記にまとめ、伏せ字も統一できます。
  雛形 `data/mask_dict.sample.yaml` をコピーして実値を入れてください。

- **除外リスト** `data/mask_allowlist.yaml`
  マスク「しない」語の名簿。一致した候補を「除外」に落とします。
  **守るのは辞書（名簿）だけ**で、連絡先の誤検出（`20181210112500@MH01R2.sdf` 型など）は外せます。
  UI の 🚫 除外リスト タブ、またはマスキング画面の「除外」操作で追加できます。

- **キャッシュ** `data/cache.db`（SQLite・自動生成）
  後述の解析キャッシュ。🗂 キャッシュ画面で一覧・削除できます。

---

## 解析キャッシュ（速度）

解析は **NER 層（GiNZA 2 モデル＝重い）** と **マスキング層（辞書照合・確信度づけ＝軽い）** に分かれます。
本ツールは **NER 層だけをキャッシュ**し、マスキング層は毎回再計算します。

- キーは「内容ハッシュ × モデル × 平文化」。**未確定でも解析時に自動保存**されます。
- 同じ文書を再解析すると NER をスキップして一瞬で終わります。
- **辞書・除外リストを変えても再 NER は不要**（軽い層だけ再計算）。
- 入力方法の **「🗂 キャッシュから選択」** で、保存済み文書をそのまま入力に再利用できます。

> 補足: `src/masking` などの自作モジュールを編集したときは、Streamlit を**再起動**してください
> （`app.py` 以外はホットリロードされません）。

---

## 検出ロジックの要点

- **候補生成**: マスク辞書 ∪ Sudachi 品詞 ∪ NER 2 モデル ∪ 連絡先の正規表現。
- **確信度**（カテゴリ別の独立チャネル数で合議）:
  - **確定** … 実辞書（名簿）一致のみ。自動マスク。
  - **強** … 2 チャネル一致／昇格／連絡先の正規表現一致。自動マスク。
  - **中** … 単独チャネル。要レビュー。
  - **弱** … 地名・その他。要レビュー。
  - **微弱** … コードらしき誤検出（`Em_NoYes` / `~C02` / `7-410` / 漢字以外の 1 文字 など）。既定で非表示。
  - **除外** … 除外リスト一致。既定で非表示。
- **自動マスク対象は 確定／強**。中・弱はレビュー、微弱・除外は確信度フィルタで既定非表示。
- マスクは原文へ当て、表記ゆれは同じ伏せ字に寄せます。

---

## アーキテクチャ（エンジンと表示層の分離）

エンジン（UI 非依存）と、表示層（CLI / Streamlit）・入力アダプタを分離しています。
エンジンはライブラリとして再利用できます。

```
src/
  masking/             ← マスキングエンジン（UI 非依存）
    engine.py            MaskingEngine（候補生成→確信度→マスク適用）
    dictionary.py        MaskDictionary（社名・商標・人名の名簿）
    allowlist.py         MaskAllowlist（除外リスト）
    cache.py             NerCache（NER 層キャッシュ・文書インデックス／SQLite）
  ner/                 ← NER エンジン（UI 非依存）
    engine.py            NerEngine / Entity / ExtractionResult
    preprocess.py        テーブル平文化（検出用）
    rendering.py         displaCy の色マップ・HTML 生成
  sources/             ← 入力アダプタ（チャンクのリストを返す）
    files.py             ファイル → チャンク（DocumentLoader + Splitter）
    kb_mcp.py            kb-mcp からの取得（分割済みチャンクをそのまま使う）
  core/document/       ← テキスト変換＋チャンク分割（kb-mcp から移植）
  config.py            ← ChunkingConfig（チャンクサイズ設定）
main.py / app.py       ← 薄い表示層（CLI シム / Streamlit UI）
```

---

## チャンク分割について（長文対策）

GiNZA 内部の SudachiPy は **1 回の解析で 49,149 バイト（≒16,000 文字弱）まで**しか扱えず、
長文を丸ごと渡すと `SudachiError: Input is too long` で落ちます。

そこで解析前に `SemanticRAGTextSplitter`
（[src/core/document/text_splitter.py](src/core/document/text_splitter.py)）で
**ファイルタイプ別にチャンク分割**し、各チャンクの結果を文字位置補正してマージします。
kb-mcp 経由の文書は格納時に分割済みなので、結合せずそのまま使います。

---

## ファイルのテキスト変換

`DocumentLoader`（[src/core/document/document_loader.py](src/core/document/document_loader.py)）が
拡張子ごとに最適なローダーへ振り分けます（kb-mcp から移植）。

| 拡張子 | ローダー | 備考 |
| --- | --- | --- |
| `.txt`, `.md` | CustomTextLoader | UTF-8 / Shift-JIS 等を自動判定 |
| `.pdf` | PdfLoader | pdfminer.six で日本語 PDF の文字化けを回避 |
| `.docx` | WordToMarkdownLoader | 見出し・表・リストを Markdown 化 |
| `.xlsx`, `.xlsm`, `.xls` | ExcelToMarkdownLoader | 各シートを Markdown テーブル化 |
| `.pptx` | PowerPointLoader | スライド・表・ノートを抽出 |
| `.html`, `.xml` | Unstructured*Loader | 別途 `uv add unstructured` が必要 |

---

## 表（テーブル）の扱い

GiNZA は自然文で学習しているため、Markdown のテーブル記法（`|` 区切り）をそのまま渡すと、
セル内の語を取りこぼします。

そこで検出のときだけ `|` を句読点に直して平文化し（[src/ner/preprocess.py](src/ner/preprocess.py)）、
**マスクは `|` 入りの原文に当てて体裁を保持**します（平文化は検出専用の内部処理です）。
