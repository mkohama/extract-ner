# =============================================================================
# data-redactor Makefile
# Docker コンテナ管理コマンド
#
# UID/GID をホストに合わせてボリューム（./data）の所有権を一致させる。
# Windows で make を使う場合は Git Bash / WSL 等の POSIX シェルから実行すること
# （`id -u` を使うため）。
# =============================================================================

.PHONY: help docker-build docker-up docker-down docker-logs clean clean-images

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
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-up       Start container (build + detached)"
	@echo "  make docker-down     Stop and remove container"
	@echo "  make docker-logs     View container logs"
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

docker-up:
	@echo "Starting data-redactor UI..."
	@mkdir -p ./data
	env UID=$(HOST_UID) GID=$(HOST_GID) docker compose up -d --build
	@echo ""
	@echo "  UI: http://localhost:8501"

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
