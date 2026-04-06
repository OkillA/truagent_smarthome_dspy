# Smart Home Configuration Agent

A config-driven conversational agent for smart home setup recommendations.

The project combines:

- CSV-defined state, operators, rules, prompts, DSPy demos, and eval cases
- generated Pydantic schemas for structured extraction
- a SOAR-inspired cognitive loop for operator selection and slot-state updates
- deterministic tools for rule evaluation
- Prometheus, Grafana, Pushgateway, and Langfuse-style tracing hooks for observability

The long-term goal is to make the system as spec-driven as possible so that changing the CSVs changes the task behavior without rewriting core runtime code.

## Current Status

The repo is no longer just a rough prototype. It now includes:

- installable Python packaging via `pyproject.toml`
- local Python pinning with `.python-version`
- config validation before generation/runtime
- generated conversation models from spec
- spec-backed DSPy examples and eval cases
- a working end-to-end agent flow for lighting, climate, and security
- test coverage for generation, validation, decoder config, evals, and engine behavior
- a full diagnostic Grafana dashboard and a separate demo dashboard

## Architecture

### 1. Config Layer

The main source of truth lives in [`agent_config/`](./agent_config):

- `00_agent_config.csv`: global runtime settings
- `01_intent_tree.csv`: slot/state definitions
- `02_action_space.csv`: operators and flow logic
- `03_tribal_knowledge.csv`: recommendation rules
- `04_utterance_decoder.csv`: extraction schema instructions
- `05_utterance_encoder.csv`: response templates
- `06_dspy_examples.csv`: few-shot DSPy examples
- `07_dspy_eval_cases.csv`: decoder eval cases

### 2. Generator Layer

[`generators/`](./generators) parses config, validates it, and generates Python artifacts:

- `csv_parser.py`: loads all spec files
- `config_validator.py`: catches broken references before runtime
- `generate_models.py`: builds generated Pydantic extraction models
- `generate_all.py`: runs the generation pipeline

Generated models are written to [`src/generated/conversation_models.py`](./src/generated/conversation_models.py).

### 3. Runtime Layer

The runtime lives in [`src/`](./src):

- [`src/soar/controller.py`](./src/soar/controller.py): Soar-like cycle coordinator around the compatibility engine
- [`src/soar/io_manager.py`](./src/soar/io_manager.py): input/output boundary for user turns and agent messages
- [`src/engine/cognitive_engine.py`](./src/engine/cognitive_engine.py): compatibility runtime preserving the current CLI behavior
- [`src/engine/control.py`](./src/engine/control.py): explicit execution control state projected into working memory
- [`src/engine/goals.py`](./src/engine/goals.py): current goal, subgoal, and impasse depth projection into working memory
- [`src/engine/operator_handler.py`](./src/engine/operator_handler.py): generic execution layer for orchestration, NLU, and action operators
- [`src/engine/runtime_model.py`](./src/engine/runtime_model.py): config-derived phase ordering, goal labels, and intent-domain routing
- [`src/engine/working_memory.py`](./src/engine/working_memory.py): WME-backed working memory with slot projection
- [`src/engine/productions.py`](./src/engine/productions.py): production compilation, matching, and preference resolution
- [`src/engine/impasse.py`](./src/engine/impasse.py): explicit impasse and substate tracking
- [`src/engine/tracing.py`](./src/engine/tracing.py): structured trace event recording
- [`src/conversation/decoder.py`](./src/conversation/decoder.py): DSPy-based structured extraction
- [`src/conversation/classifier_pipeline.py`](./src/conversation/classifier_pipeline.py): decoder prompt/schema/parsing/validation helpers
- [`src/conversation/encoder.py`](./src/conversation/encoder.py): template-based NLG
- [`src/tools/plugins/rule_evaluator.py`](./src/tools/plugins/rule_evaluator.py): deterministic recommendation engine
- [`src/ui/agent_runner.py`](./src/ui/agent_runner.py): UI-agnostic session runner
- [`src/ui/cli.py`](./src/ui/cli.py): thin terminal entrypoint

### 4. Evaluation and Observability

- [`src/evaluation/decoder_eval.py`](./src/evaluation/decoder_eval.py): runs spec-backed decoder eval cases
- Prometheus metrics across decoder, engine, tools, and session lifecycle
- Pushgateway support for preserving CLI session metrics
- Grafana dashboards in [`grafana/dashboards/`](./grafana/dashboards)

There are two dashboards:

- `LLM Observability`: full diagnostic board
- `LLM Observability Demo`: simpler shareable dashboard for standups/demos

## What The Agent Does

The agent can:

- classify whether a user is asking for a smart home task, help, off-topic response, or exit
- extract task parameters from one-shot or multi-turn conversations
- avoid re-asking questions when the user already gave enough detail
- recommend a smart home configuration based on spec-defined rules
- recover from some messy/off-topic turns during collection

### Example One-Shot Input

`I want to set up lighting in my bedroom at a low cost that is sensor based`

The agent can extract:

- intent: `configure-lighting`
- room: `bedroom`
- budget: `low`
- automation level: `reactive`

Then it can go directly to confirmation and recommendation without re-asking every field.

## Running The Project

### Install

```bash
./venv/bin/python -m pip install -e '.[dev]'
```

### Validate Config

```bash
./venv/bin/python -m generators.config_validator
```

or:

```bash
validate-smarthome-config
```

### Run The Agent

```bash
./venv/bin/python -m src.ui.cli
```

### Run Tests

```bash
./venv/bin/python -m pytest -q
```

### Run Decoder Eval

```bash
./venv/bin/python -m src.evaluation.decoder_eval
```

or:

```bash
evaluate-smarthome-decoder
```

## Observability

Bring up the monitoring stack from [`grafana/docker-compose.yml`](./grafana/docker-compose.yml):

```bash
docker compose -f grafana/docker-compose.yml up -d
```

This starts:

- Prometheus
- Grafana
- Pushgateway

Why Pushgateway is used:

- the agent currently runs as a short-lived CLI process
- Pushgateway preserves per-session metrics after the CLI exits
- Grafana can then show accumulated session metrics instead of missing short scrape windows

## Key Metrics Currently Implemented

The runtime adapts SOAR-inspired research metrics to this repo's actual architecture.
It does not expose synthetic kernel-only metrics for subsystems that do not exist here.

### Outcome / UX

- session success rate
- one-shot success rate
- turns per session
- session duration
- clarification rate
- fallback recommendation rate

### Neural-Symbolic Health

- decision cycle time
- no-change/stuck turns
- dynamic NLU requests per session
- candidate operator count
- rule no-match rate
- traceable decision rate
- handshake failures
- working-memory slot and payload size
- working-memory churn and peak known-slot growth
- cycles since last tool execution
- cycles since last meaningful progress
- constraint violation counters for invalid values, single-slot overwrites, and phase regression
- structured causal traces for recommendation decisions

### Runtime / Tooling

- tool execution counts and latency
- operator selection counts and latency
- slot update counts
- impasse metrics
- decoder scope mix
- LLM latency and extraction metrics
- NLG template generation latency
- prompt and completion token usage
- estimated LLM cost per session

## Metric Semantics

| Requested Research Metric | Repo Metric / Status |
| --- | --- |
| Decision Cycle Time | Implemented as `agent_decision_cycle_seconds` |
| WME Count | Adapted as working-memory slot counts, payload bytes, churn, and peak known-slot growth |
| Extraction Strategy / Output Link Velocity | Adapted as cycles since last tool execution and cycles since last progress |
| Constraint Violation Rate | Implemented as `agent_constraint_violation_total` |
| Faithfulness / Causal Tracing | Implemented as structured rule-evaluator metadata plus `agent_causal_trace_total` and `agent_decision_trace_total` |
| NLG Detokenization Time | Implemented as `agent_nlg_generation_seconds` for template rendering |
| Cost per Run | Implemented as prompt/completion token totals plus `agent_session_estimated_cost_usd` |
| Rete Match Time | Deferred; no Rete matcher in this runtime |
| SQLite Footprint / SMEM / EpMem | Deferred; no SQLite-backed memory stores in this runtime |
| Goal Stack Depth / Chunking / Retrieval Precision | Deferred; no substate stack, rule learning, or SMEM retrieval layer in this runtime |
| GPU VRAM / TPS / Queue Depth | Deferred; current runtime uses remote inference in a single-session CLI |

## Spec-Driven Progress

The system is partially spec-driven today, not fully.

Already spec-driven:

- slot/state definitions
- action space / operator flow
- recommendation rules
- utterance templates
- extraction schema inputs
- DSPy demos
- decoder eval cases
- some runtime behavior switches from agent config

Still primarily code-driven:

- operator execution semantics (`nlu`, `action`, `orchestration`)
- some interruption/resume logic
- overall cognitive loop structure
- dashboard/query composition

## Known Gaps

The biggest remaining areas to improve are:

- pushing more orchestration behavior into spec
- making rule explainability more structured with explicit rule ids
- further hardening interruption recovery in messy conversations
- eventually moving from a short-lived CLI runtime to a longer-lived service model

The concrete remaining path toward a stronger Soar-style runtime is documented in [`SOAR_MIGRATION.md`](./SOAR_MIGRATION.md).

## Extra Notes

Broader cleanup recommendations are documented in [`CODEBASE_RECOMMENDATIONS.md`](./CODEBASE_RECOMMENDATIONS.md).
