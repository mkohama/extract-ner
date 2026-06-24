# data-redactor

data-redactor は GiNZA ベースの**日本語ドキュメント機密マスキング**ツールです。

LLM へ渡す前の前処理として、人名・社名・商標・連絡先（メール等）などの機密情報を検出し、
伏せ字（`[社1]` など）に置き換えることを目標にしています。

検出は複数チャネルの合議で行います。チャネルは独立していて、**走ったチャネルだけを集約**します。

- **ルールベース（常時）**: マスク辞書（社名・商標・人名の名簿）＋ 連絡先の正規表現（メール）。
- **NER（任意・重い）**: Sudachi 品詞 ＋ GiNZA NER（`ja_ginza_electra` と `ja_ginza` の 2 モデル）。
- **LLM（任意・要 `az login`）**: pii-masker（Azure OpenAI `gpt-4.1-mini`）による文脈判定。
  人名/地名/外国人名/誤記など、辞書・NER で詰めきれない**文脈依存**の検出が得意。

そのうえで確信度に応じて「自動マスク／要レビュー／非表示」に振り分けます。
**recall 最優先**（マスク漏れ＝漏洩）の設計で、不確実なものは伏せずに人のレビューへ回します。

> LLM は任意です。辞書＋正規表現＋NER だけでも動きます（その場合 LLM のセットアップは不要）。

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

これで **辞書＋正規表現＋NER** のマスキングは動きます（LLM は不要）。

### LLM 検出を使う場合（任意・実機）

LLM 検出は **pii-masker**（別リポジトリ）を git submodule として取り込み、Azure OpenAI `gpt-4.1-mini` を呼びます。
`uv sync` だけでは動きません。次の 4 つが必要です。

```powershell
# 1) pii-masker のソースを external/pii-masker に取得（git submodule）
git submodule update --init
#    ※新規 clone なら `git clone --recurse-submodules <url>` で 1 と同時に取得できます

# 2) 依存をインストール（openai / azure-identity / pydantic などが入る）
uv sync

# 3) .env に Azure リソースを設定（.env.example をコピーして実値を入れる）
#    RESOURCE_NAME_GPT41_MINI=<Azure リソース名>
#    DEFAULT_LLM_MODEL=gpt-4.1-mini
#    ※ pii-masker は呼び出し元（data-redactor）の .env を読むので、**ここ**に置きます

# 4) Azure 認証（DefaultAzureCredential が使う）
az login
```

仕組み（B2 方式）: pii-masker は `[build-system]` を持たない PoC なので pip インストールせず、
`src/llm/_paths.py` が `external/pii-masker/src` を `sys.path` に通して `import pii_masker` を解決します
（submodule 未取得なら自動でスキップ＝LLM 無しで動作）。LLM 検出の本体（プロンプト・Azure・locate）は
pii-masker 側にあり、data-redactor は薄いアダプタ（[src/llm/](src/llm/)）から呼ぶだけです。

> LLM のみ実機が別環境のとき: 本物の対象データ・`data/cache.db` は実機にしかありません。
> このリポジトリ側（開発機）でも仕組みの動作確認はできますが、Azure 実呼び出しには `az login` と
> `RESOURCE_NAME_GPT41_MINI` が必要です。

#### detector_version の運用ルール（キャッシュ無効化）

LLM 検出キャッシュは `(content_hash, model, flatten, detector_version)` をキーにします。
**検出結果に影響する設定を変えたら、`app.py` の `_DETECTOR_VERSION` の対応する部分を上げる**——
こうするとキャッシュが不一致になり自動で再検出されます（上げ忘れると古い結果が使い回される＝最大の落とし穴）。

`_DETECTOR_VERSION`（例 `pii-masker@9d9942e|win7000ov200|ene-v1`）は **独立した 3 つの版**を持ち、
**上げる契機もそれぞれ別**です:

| 部分 | 上げる契機 | 担当 |
|---|---|---|
| `pii-masker@<hash>` | pii-masker（submodule）を更新したとき | `sync-pii-masker` が自動 |
| `ene-vN` | `src/masking/engine.py` の `_ENE_TO_CATEGORY`（ENE type→カテゴリ）を変えたとき | 手動 |
| `win…` | 窓ポリシー（`src/llm/windows.py` の `DEFAULT_MAX_TOKENS` / `DEFAULT_OVERLAP_TOKENS`）を変えたとき | 手動・**pii-masker とは無関係** |

> `win…`・`ene-vN` は pii-masker の更新有無に関係なく、**こちらが windows.py / engine.py を編集したとき**に
> その場で上げる作業です（下の pii-masker 追従手順とは別物）。逆に pii-masker を更新しても windows.py を
> 触っていなければ `win…` は変えません。

#### pii-masker が更新されたら（追従手順）

pii-masker（submodule）を更新するときの手順。それを取り込み、LLM 検出キャッシュを正しく無効化します。
機械的な部分は **`sync-pii-masker` サブコマンド**が自動化します。

```powershell
# 追跡ブランチの最新へ（特定のコミット/タグにするなら: data-redactor sync-pii-masker <ref>）
uv run data-redactor sync-pii-masker
```

自動で実行されること:

1. submodule のポインタを更新（`<ref>` 省略時は追跡ブランチの最新）
2. 新 HEAD の短縮ハッシュを取得
3. `app.py` の `_DETECTOR_VERSION` の `pii-masker@<hash>` を書き換え（= LLM 検出キャッシュが
   `(content_hash, model, flatten, detector_version)` 不一致で**自動ミス→再取得**になる。ここを忘れると
   検出器が変わっても古いキャッシュが使い回される＝最大の落とし穴）
4. **ENE type ドリフト検査**（pii-masker のプロンプトの型 vs `src/masking/engine.py` の
   `_ENE_TO_CATEGORY`）。マップに無い新 type は「その他」に落ちて recall 漏れになるため警告する
5. submodule の変更点（`detector_llm.py` / `schema.py` / `locate.py` 等）を表示
6. `external/pii-masker` と `app.py` を **stage**（コミットはしない）
7. `ruff` / `mypy` / `pytest` を実行

自動化できない（**人手で確認してからコミット**する）部分:

- インターフェース契約の変更（`detect` / `locate_all` の戻り値）→ [src/llm/](src/llm/) のアダプタを修正
- 新しい ENE type が増えていたら → `_ENE_TO_CATEGORY` に追加し、`ene-vN` を上げる（上の運用ルール）
- 実機（`az login` 済み）で 🤖 LLM検出 を回して件数/カテゴリを目視
- 問題なければ `git commit`

> 窓ポリシー（`win…`）は pii-masker 更新では通常触りません。変えるのは windows.py を編集したときで、
> その手順は上の「detector_version の運用ルール」を参照。

> `--no-update`（更新せず現在の HEAD で検査・検証だけ）、`--skip-tests`（ruff/mypy/pytest を省略）も使えます。

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

### マスキングの流れ（1ソース＝1パイプライン）

マスキング画面は「入力ソースを 1 つ選び、パイプラインの各ステージをタブで覗く」構成です。

1. 入力方法（✏️ テキスト / 📄 ファイル / 📚 kb-mcp / 🗂 キャッシュから選択）を選び、**[📥 読み込む]** を押す
   （チャンクを確定するだけ。重い解析はまだ走りません）。
2. 状態ヘッダーと 4 タブが出ます。**各タブが独立した実行ボタン**を持ちます。
   - **📄 平文** … テキスト化結果。
   - **🔍 NER検出** … **[▶ NER 解析を実行]**（GiNZA 2 モデル。重い）。NER 由来候補の独立ビュー。
   - **🤖 LLM検出** … **[▶ LLM 検出を実行]**（pii-masker / `gpt-4.1-mini`。要 `az login`）。LLM 単独の結果（出口1）。
   - **🔒 マージ&確信度** … **[▶ マージ&確信度を実行]**。**辞書＋正規表現（常時）＋実行済みのチャネル**
     （NER・LLM）を集約して候補化（出口2）。**GiNZA は NER 検出を実行したときだけ回ります**
     （未実行なら辞書＋regex＋LLM で軽く完結）。
3. マージ&確信度タブの候補表で確信度フィルタをかけつつチェックでマスク対象を選び、**[✅ マスクを反映]**。
   候補表の `ja_ginza` / `electra` / `Sudachi` / `LLM` / `辞書` 列で、どのチャネルが投票したか分かります。
   誤検出は「除外」→ **[🚫 選択を除外リストへ]** で以後どの文書でも候補外にできます。
4. 結果は「色付き／マスク済み／原文」で確認・ダウンロードできます。

確信度フィルタの既定は 確定・強・中・弱（**微弱・除外は既定で非表示**）。
表（`|` 区切り）は検出のときだけ平文化し、**マスクは `|` 入りの原文に当てて体裁を保持**します。

### kb-mcp 連携

「📚 kb-mcp から選択」を使うには、kb-mcp を HTTP サーバとして起動しておきます。

```powershell
# kb-mcp プロジェクト側で
uv run kb-mcp-server --transport http --port 8000
```

UI で URL（既定 `http://localhost:8000/mcp`）を指定し、「文書リストを取得」→ 文書を選択 →「📥 読み込む」。
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

> LLM 検出は現状 **UI（🤖 LLM検出 タブ）のみ**で、CLI の `mask` は 辞書＋正規表現＋NER で動きます。

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

- **候補生成（チャネル）**: マスク辞書 ∪ 連絡先の正規表現（常時）∪ Sudachi 品詞 ∪ NER 2 モデル（NER 実行時）
  ∪ LLM（pii-masker。実行時）。**走ったチャネルだけを集約**します（GiNZA は NER 検出を実行したときだけ）。
- **確信度**（解決カテゴリへ投票した独立チャネル数で合議）:
  - **確定** … 実辞書（名簿）一致のみ。自動マスク。
  - **強** … 2 チャネル一致／昇格／連絡先の正規表現一致。自動マスク。
  - **中** … 単独チャネル（LLM 単独など）。要レビュー。
  - **弱** … 地名・その他。要レビュー。
  - **微弱** … コードらしき誤検出（`Em_NoYes` / `~C02` / `7-410` / 漢字以外の 1 文字 など）。既定で非表示。
    ただし **LLM が識別子（社員番号/アカウント/IP）と判定したものは免除**（弱で残す＝レビュー可視）。
  - **除外** … 除外リスト一致。既定で非表示。
- **自動マスク対象は 確定／強**。中・弱はレビュー、微弱・除外は確信度フィルタで既定非表示。
- **LLM は「文脈を読む 1 票」**として合流します（単独→中＝レビュー／NER と相乗り→強）。
  確定は名簿のみで、LLM 単独で自動マスクはしません（過剰マスク回避）。
- マスクは原文へ当て、表記ゆれは同じ伏せ字に寄せます。

---

## アーキテクチャ（エンジンと表示層の分離）

エンジン（UI 非依存）と、表示層（CLI / Streamlit）・入力アダプタを分離しています。
エンジンはライブラリとして再利用できます。

```
src/
  masking/             ← マスキングエンジン（UI 非依存）
    engine.py            MaskingEngine（候補生成→確信度→マスク適用。analyze(run_ner=...) で NER 任意）
    dictionary.py        MaskDictionary（社名・商標・人名の名簿）
    allowlist.py         MaskAllowlist（除外リスト）
    cache.py             NerCache（NER 層 + LLM 検出層キャッシュ・文書インデックス／SQLite）
  ner/                 ← NER エンジン（UI 非依存）
    engine.py            NerEngine / sudachi_analyze_chunks（GiNZA 抜きの軽量トークナイズ）
    preprocess.py        テーブル平文化＋ build_body（spaCy 非依存の本文/オフセット構築）
    rendering.py         displaCy の色マップ・HTML 生成
  llm/                 ← LLM 検出アダプタ（pii-masker を呼ぶ薄い層。任意）
    detect_layer.py      Stage A: 窓化→pii_masker.detect/locate_all→全文スパン／cached_detect
    windows.py           本文を ~6-8k トークン窓に分割
    schema.py            LlmSpan / LlmDetection（(de)シリアライズ）
    _paths.py            external/pii-masker/src を sys.path へ（submodule path-injection）
  sources/             ← 入力アダプタ（チャンクのリストを返す）
    files.py             ファイル → チャンク（DocumentLoader + Splitter）
    kb_mcp.py            kb-mcp からの取得（分割済みチャンクをそのまま使う）
  core/document/       ← テキスト変換＋チャンク分割（kb-mcp から移植）
  config.py            ← ChunkingConfig（チャンクサイズ設定）
external/pii-masker/   ← git submodule（LLM 検出の本体。コピーせず参照）
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
