# data-redactor

GiNZA (spaCy) ベースの**日本語ドキュメント機密マスキング**ツール。LLM へ渡す前の前処理として、
人名・社名・商標・連絡先（メール等）などの機密情報を検出して伏せ字（`[社1]` 等）に置き換えます。

検出は **マスク辞書（名簿）＋ Sudachi 品詞 ＋ GiNZA NER 2モデル ＋ 正規表現（連絡先）** の合議で行い、
確信度に応じて「自動マスク／要レビュー／非表示」に振り分けます。**recall 最優先**（マスク漏れ＝漏洩）で、
不確実なものはレビューに回す設計です。NER 単体の可視化（displaCy）も付属します。

`.txt` / `.md` / `.pdf` / `.docx` / `.xlsx` / `.pptx` などのファイル、貼り付けテキスト、kb-mcp の文書を
入力にできます（kb-mcp から移植した `DocumentLoader` でテキスト化 → チャンク分割 → 解析）。

> 設計・経緯の詳細はローカルの `docs-dev/`（git 管理外）と [CLAUDE.md](CLAUDE.md) を参照。

## セットアップ

```powershell
uv sync
```

Python 3.11 / spaCy 3.7 系 / numpy 1.x に固定しています（ja_ginza_electra の依存と GiNZA 5.2 の制約のため。
3.12 や spaCy 3.8・numpy 2 では動作しません）。

## Web UI (Streamlit)

```powershell
uv run data-redactor ui
```

ブラウザで http://localhost:8501 が開きます。上部のモードで 4 つの画面を切り替えます。

- **🔒 マスキング**: 入力（✏️テキスト / 📄ファイル / 📚kb-mcp）を選び **[🔍 解析する]** を押すと候補を検出。
  確信度（確定/強は自動マスク ON、中/弱はレビュー、微弱/除外は既定で非表示）でフィルタしながら、
  チェックでマスク対象を選びます。結果は色付き / マスク済み / 原文 で確認・ダウンロードできます。
  表（`|` 区切り）は検出だけ平文化し、**マスクは `|` 入り原文に当てて体裁を保持**します。
- **🔍 固有表現抽出 (NER)**: GiNZA の固有表現を displaCy で色付き表示（検出の素を見る用）。
- **📒 マスク辞書**: 確定マスクする社名・商標・社員名の名簿（`data/mask_dict.yaml`）を編集・保存。
- **🚫 除外リスト**: マスク**しない**語（誤検出の社内コード・変数名など）の名簿（`data/mask_allowlist.yaml`）を編集。
  登録すると以後どの文書でも候補が「除外」へ落ちます（辞書・連絡先＝確定は上書きしません＝recall 安全）。

解析には GiNZA 2 モデルを使うため時間がかかります。進捗はステージ表示、所要時間は結果上部に表示します。

### kb-mcp 連携

「📚 kb-mcp から選択」を使うには kb-mcp を HTTP サーバとして起動しておきます:

```powershell
# kb-mcp プロジェクト側で
uv run kb-mcp-server --transport http --port 8000
```

UI で URL（既定 `http://localhost:8000/mcp`）を指定 →「文書リストを取得」→ 文書を選択 →「解析する」。
本文は**チャンク単位**で取得します（kb-mcp は格納時に分割済みなので結合せず解析）。

## CLI

統一コマンド `data-redactor`（実体は [src/cli.py](src/cli.py)。`uv run main.py <サブコマンド>` でも可）。

```powershell
# Streamlit UI を起動
uv run data-redactor ui

# マスキング（ファイル or --text）。既定で data/mask_dict.yaml を自動読込
uv run data-redactor mask report.pdf
uv run data-redactor mask --text "銀座のSONYの由利さんに連絡: yuri@example.co.jp"
uv run data-redactor mask report.docx --out masked.txt        # マスク済みを書き出し
uv run data-redactor mask report.docx --audit                 # 候補の票分布・確信度（表層なし＝共有OK）
uv run data-redactor mask report.docx --audit-surface         # 監査に表層も付ける（機密・共有禁止）
uv run data-redactor mask report.docx --no-flatten            # 表の平文化を切る

# NER → displaCy の HTML（ner.html）。--open で既定ブラウザ表示・--serve でサーバ表示
uv run data-redactor ner report.pdf --open
uv run data-redactor ner --text "銀座のSONYの由利さん" --labels Person --labels Company

# 各トークンの Sudachi 品詞 / NER ラベルを並べて観察（recall の穴を見る）
uv run data-redactor debug report.pdf --both-models --all-tokens

# 品質ゲート（ruff + mypy）
uv run data-redactor check
```

## マスク辞書・除外リスト（ローカル専用）

機密のため `data/*.yaml` と `data/cache*` は **git 管理外**です。各マシンで用意します。

- **マスク辞書** `data/mask_dict.yaml`: 社名・商標・社員名の名簿。一致語は**確定マスク**（全出現）。
  別表記（英語↔カタカナ・略称）を 1 つの代表表記にまとめ、伏せ字も統一できます。
  雛形 `data/mask_dict.sample.yaml` をコピーして実値を入れてください。
- **除外リスト** `data/mask_allowlist.yaml`: マスクしない語。検出由来の誤検出だけを「除外」に落とし、
  辞書・連絡先（確定）は守ります。UI の 🚫 除外リスト タブで編集できます。

## 検出ロジックの要点

- **候補生成**: マスク辞書 ∪ Sudachi 品詞（人名/地名/固有名詞）∪ NER 2モデル ∪ 連絡先の正規表現。
- **確信度（カテゴリ別の独立チャネル数で合議）**:
  - **確定** = 実辞書一致 または 連絡先の正規表現（決定的）
  - **強** = 2 チャネル一致 など（自動マスク）
  - **中** = 単独チャネル（要レビュー）
  - **弱** = 地名・その他（要レビュー）
  - **微弱** = コードらしき誤検出（`Em_NoYes` / `~C02` / `7-410` / 1文字英字 など。既定で非表示）
  - **除外** = 除外リスト一致（既定で非表示）
- **自動マスク対象は 確定/強**。中/弱はレビュー、微弱/除外は確信度フィルタで既定非表示。
- マスクは原文へ当て、表記ゆれは同じ伏せ字に寄せます。

## アーキテクチャ（エンジンと表示層の分離）

エンジン（UI 非依存）と表示層（CLI / Streamlit）・入力アダプタを分離。エンジンはライブラリとして再利用できます。

```
src/
  masking/             ← マスキングエンジン（UI 非依存）
    engine.py            MaskingEngine（候補生成→確信度→マスク適用）
    dictionary.py        MaskDictionary（社名・商標・人名の名簿）
    allowlist.py         MaskAllowlist（除外リスト）
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

## チャンク分割について（長文対策）

GiNZA 内部の SudachiPy は **1 回の解析で 49,149 バイト（≒16,000 文字弱）まで**しか扱えず、
長文を丸ごと渡すと `SudachiError: Input is too long` で落ちます。そこで解析前に
`SemanticRAGTextSplitter`（[src/core/document/text_splitter.py](src/core/document/text_splitter.py)）で
**ファイルタイプ別にチャンク分割**してから解析し、各チャンクの結果を文字位置補正してマージします。
kb-mcp 経由の文書は格納時に分割済みなので結合せずそのまま使います。

## ファイルのテキスト変換

`DocumentLoader`（[src/core/document/document_loader.py](src/core/document/document_loader.py)）が拡張子ごとに
最適なローダーへ振り分けます（kb-mcp から移植）。

| 拡張子 | ローダー | 備考 |
| --- | --- | --- |
| `.txt`, `.md` | CustomTextLoader | UTF-8/Shift-JIS 等を自動判定 |
| `.pdf` | PdfLoader | pdfminer.six で日本語 PDF の文字化けを回避 |
| `.docx` | WordToMarkdownLoader | 見出し・表・リストを Markdown 化 |
| `.xlsx`, `.xlsm`, `.xls` | ExcelToMarkdownLoader | 各シートを Markdown テーブル化 |
| `.pptx` | PowerPointLoader | スライド・表・ノートを抽出 |
| `.html`, `.xml` | Unstructured*Loader | 別途 `uv add unstructured` が必要 |

## 表（テーブル）の扱い

GiNZA は自然文で学習しているため、Markdown のテーブル記法（`|` 区切り）をそのまま渡すとセル内の語を
取りこぼします。検出時のみ `|` を句読点に直して平文化し（[src/ner/preprocess.py](src/ner/preprocess.py)）、
**マスクは `|` 入り原文に当てて体裁を保持**します（平文化は検出専用の内部処理）。
