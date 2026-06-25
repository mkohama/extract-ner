# =============================================================================
# data-redactor UI Dockerfile
# Streamlit ベースの UI コンテナ（マルチステージビルド）
#
# data-redactor 固有の注意（spec-summarizer の型からの差分）:
#   - Python は 3.11 固定（ja-ginza-electra の tokenizers 制約。3.12 だと wheel 無し）。
#   - pii-masker は git submodule（pip 非インストール）。external/pii-masker/src を COPY し
#     PYTHONPATH=/app 経由の path-injection（src/llm/_paths.py）で import を解決する。
#   - app.py / main.py はルート直下スクリプト。wheel には入らない（packages=["src"]）ので
#     プロジェクトを site-packages にインストールせず（--no-install-project）、ソースツリーを
#     /app へ COPY し PYTHONPATH=/app で `import src` を解決する。これにより _paths.py の
#     parents[2] が /app を指し、/app/external/pii-masker/src を sys.path に通せる。
#   - ja_ginza_electra は初回 nlp ロードで torch＋ELECTRA 重みを DL する。ビルド時に prewarm
#     してイメージに焼く（実行時ネット不要・recall 既定を担保）。HF_HOME を固定パスに置き、
#     任意の実行 UID でも読めるよう a+rX を付ける。
# =============================================================================

# グローバル ARG（各ステージで再宣言が必要）
ARG APP_UID=1000
ARG APP_GID=1000

# -----------------------------------------------------------------------------
# Stage 1: Builder — 依存関係をインストール
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 依存のみ先にインストール（レイヤキャッシュ効率化）。本プロジェクトに optional-deps は無いので
# --extra は不要。app.py/main.py を wheel に含めないため --no-install-project（ソース直実行）。
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 不要なビルドアーティファクトを削除してイメージサイズを削減
RUN find /app/.venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pyo" -delete 2>/dev/null || true

# -----------------------------------------------------------------------------
# Stage 2: Runtime — 最終的な実行イメージ
# -----------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Azure CLI をインストール（pii-masker の Entra ID 認証で az account get-access-token を使用）
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 非 root ユーザーを作成（ホスト側の UID/GID に合わせる）
ARG APP_UID
ARG APP_GID
RUN groupadd -g ${APP_GID} appuser && \
    useradd -u ${APP_UID} -g ${APP_GID} --create-home appuser

# builder から仮想環境のみコピー（torch / spaCy / GiNZA を含む）。
# ★ --chown で所有権を付与する: 別途 `chown -R /app/.venv` を打つと venv 全体が新レイヤに複製され
#   イメージが数 GB 単位で膨らむ（COPY 時点で所有権を確定すれば追加レイヤは生じない）。
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# 環境変数
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
# HF/transformers キャッシュは appuser が所有する /app 配下に置く（root 不要・再帰 chmod 不要）。
ENV HF_HOME="/app/.cache/huggingface"

# 永続化マウント点と HF キャッシュ用ディレクトリを appuser 所有で用意（venv には触れない＝複製しない）。
RUN mkdir -p /app/data "${HF_HOME}" && chown appuser:appuser /app /app/data /app/.cache "${HF_HOME}"

USER appuser

# GiNZA electra を prewarm（torch＋ELECTRA 重みをイメージに焼く）。appuser で実行するので
# 重みは appuser 所有・既定パーミッション 644（他 UID からも読める）＝再帰 chmod は不要。
# ★ 再帰 chmod/chown は絶対に打たない: venv を丸ごと別レイヤに複製してイメージが数倍になる。
# ★ ソース COPY より前に置く: prewarm は .venv（依存）にのみ依存する。pii-masker submodule や
#   app/src を更新しても、依存が変わらない限りこの重い層はキャッシュ再利用される（再 DL されない）。
#   依存（pyproject/uv.lock）を変えたときだけ builder の uv sync ともども再実行される＝正しい挙動。
RUN python -c "import spacy; spacy.load('ja_ginza_electra')"

# アプリのソースツリーを COPY（wheel に含まれないルートスクリプトと submodule）。--chown で所有権を付与。
# pii-masker 更新で最初に invalidate されるのはここ。以降は処理が無いので再ビルドが速い。
#   - app.py / main.py        : エントリ（streamlit run app.py）
#   - src/                    : エンジン・マスキング・LLM アダプタ（config.py も src 配下）
#   - external/pii-masker/src : path-injection で import する pii-masker 本体
COPY --chown=appuser:appuser app.py main.py ./
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser external/pii-masker/src/ external/pii-masker/src/

# Streamlit 設定
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# USER は prewarm 直前で appuser に切替済み（ここでの再指定は不要）。
CMD ["streamlit", "run", "/app/app.py"]
