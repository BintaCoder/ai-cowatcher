.PHONY: up down logs install api health

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

api:
	.venv/bin/uvicorn ai_cowatcher.main:app --host 0.0.0.0 --port 8000 --reload

health:
	curl -s http://localhost:8000/health | python3 -m json.tool
