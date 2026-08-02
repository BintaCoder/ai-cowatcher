.PHONY: up down logs install api api-dev worker health

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

install:
	python3.12 -m venv .venv
	.venv/bin/python3.12 -m pip install --upgrade pip setuptools wheel
	.venv/bin/python3.12 -m pip install -r requirements.txt
	.venv/bin/python3.12 -m pip install -e .
	@echo "$(CURDIR)" > .venv/lib/python3.12/site-packages/cowatcher-dev.pth

# Stable serve: one process. --reload forks a 2nd process and can OOM-kill when
# BGE-M3 warms at startup (macOS reports "Killed: 9" / make exit).
api:
	TOKENIZERS_PARALLELISM=false .venv/bin/uvicorn ai_cowatcher.main:app --host 0.0.0.0 --port 8000

# Auto-reload for light code edits only (avoid with MOCK_MODE=false + BGE warm).
api-dev:
	TOKENIZERS_PARALLELISM=false .venv/bin/uvicorn ai_cowatcher.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	.venv/bin/cowatcher-ingest-worker

health:
	curl -s http://localhost:8000/health | python3 -m json.tool
