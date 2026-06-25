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

# builder から仮想環境のみコピー（torch / spaCy / GiNZA を含む）
COPY --from=builder /app/.venv /app/.venv

# アプリのソースツリーを COPY（wheel に含まれないルートスクリプトと submodule）。
#   - app.py / main.py        : エントリ（streamlit run app.py）
#   - src/                    : エンジン・マスキング・LLM アダプタ（config.py も src 配下）
#   - external/pii-masker/src : path-injection で import する pii-masker 本体
COPY app.py main.py ./
COPY src/ src/
COPY external/pii-masker/src/ external/pii-masker/src/

# 環境変数
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
# HuggingFace / transformers のキャッシュを固定パスに集約（任意 UID から読めるようにする）
ENV HF_HOME="/opt/hf-cache"

# GiNZA electra を prewarm（torch＋ELECTRA 重みをイメージに焼く）。spacy.load が .venv 内の
# ja_ginza_electra パッケージ配下と HF_HOME の双方へ書き込む可能性があるため、両方を後で a+rX。
RUN mkdir -p "${HF_HOME}" && \
    python -c "import spacy; spacy.load('ja_ginza_electra')" && \
    chmod -R a+rX "${HF_HOME}" /app/.venv

# データ永続化のマウント点を用意（cache.db / mask_dict.yaml / mask_allowlist.yaml）
RUN mkdir -p /app/data && chown -R appuser:appuser /app

# Streamlit 設定
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

USER appuser

CMD ["streamlit", "run", "/app/app.py"]
