# CLAUDE.md

extract-ner — GiNZA (spaCy) による日本語固有表現抽出（NER）ツール。
ファイル / テキスト / kb-mcp の文書をテキスト化・チャンク化して NER し、displaCy で表示する。

## ★ 運用ルール（最優先）

- **試行内容・気づき・うまくいかなかったことと理由は、必ず [docs-dev/insight-memo.md](docs-dev/insight-memo.md) に
  日付つきで追記する。** これはこのプロジェクトの決まり。実装で何か学んだら、コードを変えたら、
  まずここに残すこと（過去の試行錯誤がすべて時系列で蓄積されている。重複検証を避けるため必ず先に読む）。
- ドキュメント類は日本語で書く。

## 環境・依存（固定されている。安易に上げない）

- **Python 3.11 固定**（`requires-python = ">=3.11,<3.12"`）。ja-ginza-electra が古い
  tokenizers を要求するため。3.12 だと wheel が無くビルドに失敗する。
- **spaCy 3.7 系 / numpy 1.x 固定**。GiNZA 5.2 は spaCy 3.8 で起動せず、spaCy 3.7 の
  thinc バイナリは numpy 2.x と非互換。
- 動作確認済み: GiNZA 5.2.0 / ja-ginza 5.2.0 / ja-ginza-electra 5.2.0 / spaCy 3.7.5 / numpy 1.26.4。
- パッケージ管理は **uv**（`uv sync` / `uv add` / `uv run ...`）。

## 開発ワークフロー

- 実行は統一サブコマンド **`extract-ner`**（`[project.scripts]`、実体は `src/cli.py`。click ベース）:
  - `uv run extract-ner ui` … Streamlit UI を起動（`streamlit run app.py` のラッパ）
  - `uv run extract-ner ner <file> [--open/--serve/--model/--labels/--flatten]` … NER → HTML 表示
  - `uv run extract-ner debug <file> [--both-models/--all-tokens/--flatten/--out]` …
    各トークンの SudachiPy 品詞 / NER ラベルを並べて recall の穴を観察
  - `uv run extract-ner check` … 品質ゲート（ruff + mypy）をまとめて実行
  - `main.py` は後方互換シム（`uv run main.py <サブコマンド>` でも同じ）。
  - エントリポイント登録には `pyproject.toml` の `[build-system]`（hatchling, `packages=["src"]`）と
    `[project.scripts]` が必要。追加後は `uv sync` で再インストールするとコマンドが有効になる。
- 品質ゲート: `uv run extract-ner check`（= `uv run ruff check src main.py app.py` ＋
  `uv run mypy src main.py app.py`）。
  - mypy の既存エラー（`src/utils/text_utils.py` / `loaders/powerpoint_loader.py`）は kb-mcp
    移植元由来。**新規にエラーを増やさないこと**。スタブ無しライブラリは pyproject の
    `[[tool.mypy.overrides]]` に追加して抑制する（既存の langchain_* / tiktoken と同様）。
  - black は環境に入っていないことがある。スタイルは周囲に合わせる。
- **`.venv` を握ったまま再 sync / 再起動するとアクセス拒否になる**。先に streamlit を止める:
  `Get-Process | ? { $_.CommandLine -like '*streamlit*app.py*' } | Stop-Process -Force`
- Streamlit は app.py はホットリロードするが、**import した自作モジュール（main.py 等）の
  変更は確実には反映されない**。直したらサーバー再起動。

## アーキテクチャ（エンジンと表示の分離を保つ）

```
src/
  ner/                エンジン（UI 非依存・spaCy/displaCy 以外に依存しない）
    engine.py           NerEngine / Entity / ExtractionResult。extract() / extract_chunks()
    preprocess.py       テーブル平文化など
    rendering.py        displaCy の色マップ・HTML 生成
  sources/            入力アダプタ（→ チャンクのリストを返す）
    files.py            load_chunks_from_file / load_text_from_file
    kb_mcp.py           get_document_chunks_sync / list_documents_sync
  core/document/      テキスト変換＋チャンク分割（kb-mcp から移植）
    document_loader.py  拡張子別ローダーへ振り分け
    text_splitter.py    SemanticRAGTextSplitter（ファイルタイプ別チャンク分割）
    splitters/          base / default / markdown / pdf / excel / token_utils
  config.py           ChunkingConfig（チャンクサイズ設定。kb-mcp と同値）
main.py / app.py      薄い表示層（CLI / Streamlit）
```

- 表示層（main.py / app.py）はエンジンを呼ぶだけ。エンジンに Streamlit/IO を持ち込まない。

## 重要な地雷・設計判断

- **SudachiPy のトークナイズ上限 = 49,149 バイト**（文字数でなくバイト数。日本語 UTF-8 で
  約16,000文字弱）。長文を `nlp(text)` に丸ごと渡すと `SudachiError: Input is too long` で落ちる。
  → 必ず **チャンク分割してから** NER する。入力層は「テキスト1本」でなく
  **チャンクのリスト**を返し、`NerEngine.extract_chunks()` が各チャンクを `nlp.pipe` で解析して
  オフセット補正マージする。`extract(text)` は `extract_chunks([text])` のラッパ。
  保険として `_byte_safe_pieces`（engine.py）が巨大チャンクをバイト数で再分割する。
  詳細は insight-memo「2026-06-04 長文で SudachiError」参照。
- チャンク分割は kb-mcp と同じ単位（xlsx≈1000トークン/overlap0 等。tiktoken cl100k_base）。
  RAG 格納時と同じ単位 = 検索ヒット単位と抽出結果が揃う。
- **NER が効く層 / 効かない層**: 一般固有名詞（人名・地名・日付・金額）は GiNZA（特に
  electra）が得意。一方、社内ジャーゴン/専門用語（独自のマーク名・列挙子・変数名など）は
  関根の拡張固有表現体系に無く **NER 単体では引けない**。これは `EntityRuler`/辞書アプローチが必要
  （モデルや前処理のチューニングでは伸びない）。insight-memo の該当節を参照。
- モデル既定は `ja_ginza_electra`（高精度・低速、torch 必要・初回 DL あり）。
  軽量・高速は `ja_ginza`。`--model` / UI のサイドバーで切替。
- displaCy 表示は manual モード（`to_displacy_data`）で統一。色マップのキーは大文字で突き合わせる
  （GiNZA の実ラベルは Title case、displaCy が `.upper()` で引くため）。

## セッション履歴の保存場所（豆知識）

Claude Code のセッション履歴は**プロジェクトの絶対パスごと**に
`~/.claude/projects/<エンコードしたパス>/` に保存される。**リポジトリ／フォルダ名を変えると
履歴が見えなくなる**（旧パスのフォルダに残っている）。引き継ぐには旧フォルダの `*.jsonl` を
新パスのフォルダにコピーする。
