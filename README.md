# extract-ner

GiNZA (`ja_ginza`) で日本語の固有表現抽出 (NER) を実行し、spaCy の displaCy で
色付きハイライト表示する HTML を生成するプログラムです。

`.txt` / `.md` / `.pdf` / `.docx` / `.xlsx` / `.pptx` などのファイルを指定すると、
kb-mcp から移植したテキスト変換部 ([src/](src/) 配下の `DocumentLoader`) で
テキスト化してから解析します。

## セットアップ

```powershell
uv sync
```

## Web UI (Streamlit)

ドキュメントをアップロードすると、テキスト化 → NER → 色付き表示まで行う UI です。

```powershell
uv run streamlit run app.py
```

ブラウザで http://localhost:8501 が開きます。

- 左サイドバーの「⚙️ 設定」: モデル切替（既定 `ja_ginza_electra`）／テーブル平文化トグル（既定 OFF）
- 入力方法をラジオで選択:
  - **📄 ファイルをアップロード**: 対応形式のファイルをドラッグ＆ドロップ
  - **✏️ テキストを入力**: テキストを貼り付けて解析
  - **📚 kb-mcp から選択**: kb-mcp サーバの文書一覧から選んで解析（要 kb-mcp 起動）
- 「表示するラベル」マルチセレクトで、ハイライト・一覧に出すラベルを絞り込み可能（初期は全件）
- 抽出された固有表現を displaCy のハイライト表示・ラベル別件数・一覧表で確認できます

### kb-mcp 連携

「📚 kb-mcp から選択」を使うには、kb-mcp 側を HTTP サーバとして起動しておきます:

```powershell
# kb-mcp プロジェクト側で
uv run kb-mcp-server --transport http --port 8000
```

UI で URL（既定 `http://localhost:8000/mcp`）を指定し「文書リストを取得」→ 文書を選択すると、
その全文を取得して解析します（[src/kb_mcp_client.py](src/kb_mcp_client.py) は term-variants から移植）。

## CLI の使い方

```powershell
# サンプル文を解析して ner.html を生成
uv run main.py

# 生成した HTML を既定ブラウザで開く
uv run main.py --open

# 任意のテキストを直接解析
uv run main.py --text "解析したい文章"

# 任意のファイルをテキスト化して解析 (.txt/.md/.pdf/.docx/.xlsx/.pptx/.html/.xml)
uv run main.py --input report.pdf
uv run main.py --input data.xlsx --open

# モデルを切り替え (既定 ja_ginza_electra / 軽量・高速の ja_ginza)
uv run main.py --model ja_ginza --input report.pdf

# 特定カテゴリ（ラベル）だけ抽出
uv run main.py --labels Person Company --text "銀座のSONYの由利さん"

# Markdown テーブルを平文化してから解析 (既定はそのまま解析)
uv run main.py --flatten --input table.md

# ブラウザで http://localhost:5000 を開いて表示 (Ctrl+C で終了)
uv run main.py --serve
```

抽出結果はコンソールにも一覧表示され、`ner.html` に displaCy のハイライト表示を出力します。

## アーキテクチャ（エンジンと UI の分離）

固有表現抽出の中核（エンジン）と、表示層（CLI / UI）・入力ソースを分離しています。
エンジンは Streamlit や displaCy に依存しないため、ライブラリとして再利用できます。

```
src/
  ner/                 ← エンジン（UI 非依存）
    engine.py            NerEngine / Entity / ExtractionResult（カテゴリ絞り込み込み）
    preprocess.py        テーブル平文化などの前処理
    rendering.py         displaCy 用の色マップ・HTML 生成（表示ヘルパ）
  sources/             ← 入力アダプタ
    files.py             ファイル → テキスト（DocumentLoader 経由）
    kb_mcp.py            kb-mcp からの取得
    __init__.py          SAMPLE_TEXT など
  core/document/       ← テキスト変換ライブラリ（kb-mcp から移植）
main.py                ← CLI（薄い表示層）
app.py                 ← Streamlit UI（薄い表示層）
```

エンジンの使用例（テキストから抽出 / 特定カテゴリだけ抽出）:

```python
from src.ner import NerEngine

engine = NerEngine("ja_ginza_electra")

# 全カテゴリ抽出
result = engine.extract("銀座のSONYに勤める由利さん")
for ent in result.entities:
    print(ent.text, ent.label, ent.start, ent.end)

# 特定カテゴリだけ抽出（テーブル平文化も任意で）
companies = engine.extract(text, labels=["Company"], flatten_tables=True)

# 抽出済みの結果から後段で絞り込み
persons = result.filter(["Person"])
```

## ファイルのテキスト変換について

`--input` で指定したファイルは [src/core/document/document_loader.py](src/core/document/document_loader.py)
の `DocumentLoader` で拡張子ごとに最適なローダーへ振り分けてテキスト化します
(kb-mcp プロジェクトのテキスト変換部をそのまま移植)。

| 拡張子 | ローダー | 備考 |
| --- | --- | --- |
| `.txt`, `.md` | CustomTextLoader | UTF-8/Shift-JIS 等を自動判定 |
| `.pdf` | PdfLoader | pdfminer.six で日本語 PDF の文字化けを回避 |
| `.docx` | WordToMarkdownLoader | 見出し・表・リストを Markdown 化 |
| `.xlsx`, `.xlsm`, `.xls` | ExcelToMarkdownLoader | 各シートを Markdown テーブル化 |
| `.pptx` | PowerPointLoader | スライド・表・ノートを抽出 |
| `.html`, `.xml` | Unstructured*Loader | 別途 `uv add unstructured` が必要 |

`.html` / `.xml` を使う場合のみ追加インストールが必要です:

```powershell
uv add unstructured
```

## 表（テーブル）の扱い

GiNZA は自然文で学習しているため、Markdown のテーブル記法（`|` 区切り）を
そのまま渡すとセル内の語をほとんど抽出できません。そこで解析前に
`prepare_for_ner()`（[src/ner/preprocess.py](src/ner/preprocess.py)）でテーブル記法を取り除きます。

- 区切り行（`| --- | --- |`）は削除
- データ行はセルを区切り文字（既定は読点「、」、`TABLE_CELL_DELIMITER` で変更可）で
  連結し、末尾に句点を付与

例: `| 由利 | 33 | Sony |` → `由利、33、Sony。`

ヘッダー行に依存しないため、**テーブルの途中だけを含むチャンク**に適用しても
破綻せず、`| 由利` のような記号混じりの誤抽出も生じません。

ただし GiNZA の NER は文脈依存が強く、短い語や曖昧な語（例: "Sony"）は
どの整形をしても抽出されないことがあります（表データの抽出は本質的に不安定）。

## ラベルの配色

GiNZA は関根の拡張固有表現階層 (PERSON / DATE / TIME / AGE / MONEY /
POSITION_VOCATION / N_PERSON / OCCASION_OTHER など) のラベルを出力します。
[src/ner/rendering.py](src/ner/rendering.py) の `ENT_COLORS` でラベルごとの色を調整できます。

## 補足

- 依存関係の都合で Python 3.11 / spaCy 3.7 系 / numpy 1.x に固定しています
  (ja_ginza_electra が古い tokenizers を要求するため Python 3.11、
  GiNZA 5.2 と spaCy 3.8・numpy 2 の組み合わせは動作しません)。
- Windows のコンソール表示が文字化けする場合がありますが、UTF-8 で出力される
  `ner.html` は正しく表示されます。
