.PHONY: up up-core down logs install api api-dev ingest worker health bench-ask

# Full stack including brokers (Kafka/RabbitMQ). Kafka uses apache/kafka (not Bitnami).
up:
	docker compose up -d

# Core pilot deps only (skip Kafka/Rabbit if you use MESSAGE_BROKER=memory).
up-core:
	docker compose up -d postgres redis qdrant neo4j minio prometheus grafana

down:
	docker compose down

logs:
	docker compose logs -f

# Create/refresh venv + editable package install (fixes ModuleNotFoundError: ai_cowatcher).
install:
	python3.12 -m venv .venv
	.venv/bin/python3.12 -m pip install --upgrade pip setuptools wheel
	.venv/bin/python3.12 -m pip install -r requirements.txt
	.venv/bin/python3.12 -m pip install -e ".[dev]"
	@echo "$(CURDIR)" > .venv/lib/python3.12/site-packages/cowatcher-dev.pth
	@.venv/bin/python -c "import ai_cowatcher; print('ok:', ai_cowatcher.__file__)"

# Stable serve: one process. --reload forks a 2nd process and can OOM-kill when
# BGE-M3 warms at startup (macOS reports "Killed: 9" / make exit).
api:
	TOKENIZERS_PARALLELISM=false .venv/bin/uvicorn ai_cowatcher.main:app --host 0.0.0.0 --port 8000

# Auto-reload for light code edits only (avoid with MOCK_MODE=false + BGE warm).
api-dev:
	TOKENIZERS_PARALLELISM=false .venv/bin/uvicorn ai_cowatcher.main:app --host 0.0.0.0 --port 8000 --reload

# Offline ingest via the venv interpreter (most reliable; does not depend on PATH scripts).
# Usage:
#   make ingest TITLE=friends_ross VIDEO=friends_ross_has_problems.mp4
#   make ingest TITLE=friends_ross VIDEO=./friends_ross_has_problems.mp4 FORCE=1
ingest:
	@test -n "$(TITLE)" || (echo 'Set TITLE=... e.g. TITLE=friends_ross' && exit 1)
	@test -n "$(VIDEO)" || (echo 'Set VIDEO=... path to mp4' && exit 1)
	TOKENIZERS_PARALLELISM=false .venv/bin/python -m ai_cowatcher.ingestion.cli \
		--title-id "$(TITLE)" --video "$(VIDEO)" $(if $(FORCE),--force,)

worker:
	.venv/bin/python -m ai_cowatcher.ingestion.worker_cli

health:
	curl -s http://localhost:8000/health | python3 -m json.tool

# Sample 5 random playhead questions against live /ask (real Gemini; MOCK_MODE=false).
# Results → Postgres bench_ask_result + benchmarks/results/<run_id>.jsonl → Grafana "Ask Bench".
# Usage: make bench-ask
#        make bench-ask SEED=42 N=5 TITLE="Friends Ross"
#        make bench-ask PERSONA=witty_friend GENDER=female
#        make bench-ask ALL_PERSONAS=1 N=3   # same samples × each persona
bench-ask:
	TOKENIZERS_PARALLELISM=false PYTHONPATH=. .venv/bin/python -m ai_cowatcher.bench.ask_runner \
		--title-id "$(or $(TITLE),Friends Ross)" \
		--n "$(or $(N),5)" \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(BASE_URL),--base-url $(BASE_URL),) \
		$(if $(DURATION),--duration-sec $(DURATION),) \
		$(if $(PERSONA),--persona-id $(PERSONA),) \
		$(if $(GENDER),--companion-gender $(GENDER),) \
		$(if $(ALL_PERSONAS),--all-personas,) \
		$(if $(ALLOW_MOCK),--allow-mock,)
