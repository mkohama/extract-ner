# =============================================================================
# data-redactor Makefile
# Docker コンテナ管理コマンド
#
# UID/GID をホストに合わせてボリューム（./data）の所有権を一致させる。
# Windows で make を使う場合は Git Bash / WSL 等の POSIX シェルから実行すること
# （`id -u` を使うため）。
# =============================================================================

.PHONY: help docker-build docker-sync-build docker-up docker-down docker-logs clean clean-images

.DEFAULT_GOAL := help

# Host user ID / group ID（Docker ボリュームの権限合わせ）
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)

# =============================================================================
# Help
# =============================================================================
help:
	@echo ""
	@echo "=== data-redactor Docker Commands ==="
	@echo ""
	@echo "  make docker-build       Build Docker image"
	@echo "  make docker-sync-build  Bump pii-masker submodule (venv-free) + build"
	@echo "                          (optional: PII_REF=<commit/tag/branch>)"
	@echo "  make docker-up          Start container (build + detached)"
	@echo "  make docker-down        Stop and remove container"
	@echo "  make docker-logs        View container logs"
	@echo ""
	@echo "=== Maintenance ==="
	@echo ""
	@echo "  make clean           Remove container and volumes"
	@echo "  make clean-images    Remove data-redactor image"
	@echo ""

# =============================================================================
# Docker Compose
# =============================================================================
docker-build:
	@echo "Building Docker image..."
	env UID=$(HOST_UID) GID=$(HOST_GID) docker compose build

# pii-masker（submodule）を取り込んでからイメージを再ビルドする（ビルド/配布機向け）。
# ★ ホスト側 .venv を作らない: sync-pii-masker（uv run）と違い git + perl だけで機械的に済ませる。
#    イメージに効くのは 2 つ (a) external/pii-masker/src の更新 (b) app.py の pii-masker@<hash> だけで、
#    どちらも venv 不要。ENE ドリフト検査 / ruff / mypy / pytest は開発機の `sync-pii-masker` に委ねる。
#    置換は perl（Git Bash 同梱）で行う: sed -i は CRLF を LF に潰すが perl -i -pe は改行を保持する。
# 既定は追跡ブランチの最新へ更新。特定の版に固定するなら: make docker-sync-build PII_REF=<commit/tag/branch>
docker-sync-build:
	@echo "Syncing pii-masker submodule (venv-free)..."
	@if [ -n "$(PII_REF)" ]; then \
		git -C external/pii-masker fetch && git -C external/pii-masker checkout "$(PII_REF)"; \
	else \
		git submodule update --remote external/pii-masker; \
	fi
	@HASH=$$(git -C external/pii-masker rev-parse --short HEAD); \
	 echo "pii-masker HEAD: $$HASH"; \
	 OLD=$$(perl -ne 'if (/pii-masker\@([0-9a-fA-F]+)/) { print $$1; last }' app.py); \
	 if [ -z "$$OLD" ]; then \
		echo "WARNING: pii-masker@<hash> not found in app.py; skipped rewrite (check _DETECTOR_STATIC)"; \
	 elif [ "$$OLD" = "$$HASH" ]; then \
		echo "app.py detector hash already latest (pii-masker@$$HASH); no change, LLM cache stays valid"; \
	 else \
		perl -i -pe "s/pii-masker\@[0-9a-fA-F]+/pii-masker\@$$HASH/" app.py; \
		echo "Rewrote app.py detector hash pii-masker@$$OLD -> pii-masker@$$HASH (LLM cache will auto-miss)"; \
	 fi
	@$(MAKE) docker-build

docker-up:
	@echo "Starting data-redactor UI..."
	@mkdir -p ./data
	env UID=$(HOST_UID) GID=$(HOST_GID) docker compose up -d --build
	@echo ""
	@echo "  UI: http://localhost:8508"

docker-down:
	@echo "Stopping data-redactor..."
	env UID=$(HOST_UID) GID=$(HOST_GID) docker compose down --remove-orphans

docker-logs:
	env UID=$(HOST_UID) GID=$(HOST_GID) docker compose logs -f

# =============================================================================
# Maintenance
# =============================================================================
clean:
	@echo "Removing container and volumes..."
	env UID=$(HOST_UID) GID=$(HOST_GID) docker compose down -v --remove-orphans
	@echo "Done."

clean-images:
	@echo "Removing data-redactor image..."
	-docker rmi data-redactor-ui:latest 2>/dev/null || true
	@echo "Done."
