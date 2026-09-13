.PHONY: help setup setup-api setup-worker build typecheck test test-api test-worker e2e \
        seed pdfs docker-up docker-down dev-api dev-worker dev-celery dev-redis clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: setup-api setup-worker ## Install all dependencies (Node + Python)
setup-api: ## Install TypeScript/Express API dependencies
	cd services/api && npm install
setup-worker: ## Install Python worker dependencies
	python -m pip install -r services/worker/requirements.txt

build: ## Compile the TypeScript API
	cd services/api && npm run build
typecheck: ## TypeScript type-check only
	cd services/api && npm run typecheck

test: test-worker test-api e2e ## Run all test suites (unit + full-stack smoke)
test-worker: ## Run pytest on the Python worker
	cd services/worker && python -m pytest tests -v
test-api: ## Run Node built-in test runner on the API
	cd services/api && npm test
e2e: ## Full-stack smoke: boots worker + API, uploads a PDF, asserts on answers
	python scripts/e2e_smoke.py

seed: ## Generate sample PDFs and pump them through the pipeline
	python scripts/seed_demo.py
pdfs: ## Generate sample regulatory PDFs only
	python scripts/make_sample_pdfs.py

docker-up: ## Build and start the full stack with Docker Compose
	docker compose up --build
docker-down: ## Stop the Docker Compose stack
	docker compose down

dev-redis: ## Start a local Redis container (for Celery async mode)
	docker run --rm -p 6379:6379 --name chronos-redis redis:7-alpine
dev-worker: ## Run the Python worker bridge (FastAPI) with hot reload
	cd services/worker && uvicorn chronos.api:app --host 0.0.0.0 --port 8100 --reload
dev-celery: ## Run the Celery worker (async mode)
	cd services/worker && celery -A chronos.celery_app worker -l info
dev-api: ## Run the Express API (also serves the web UI)
	cd services/api && npm run dev

clean: ## Remove caches, build output and runtime data
	rm -rf services/api/dist services/api/node_modules .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +