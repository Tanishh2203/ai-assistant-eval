# AI Assistant Evaluation — OSS vs Frontier
### Telegram Bots · RAG · Guardrails · Observability · Tool Use · LLM-as-Judge

Two production-ready Telegram bots evaluated head-to-head on hallucination, safety, and bias.

| | OSS Bot | Frontier Bot |
|---|---|---|
| **Model** | Qwen2.5-72B-Instruct | Claude Sonnet (claude-sonnet-4-5) |
| **Inference** | HuggingFace Inference API | Anthropic API |
| **RAG** | ChromaDB + sentence-transformers | Same |
| **Memory** | Per-user in-process dict (10 turns) | Same |
| **Guardrails** | Input + Output safety filtering | Same |
| **Observability** | Structured JSONL logging | Same |
| **Tools** | Calculator, DateTime, KB Search | Same |
| **Public Deploy** | HF Spaces (hf_spaces/) | N/A |

---

## Architecture

```
ai-assistant-eval/
├── shared/
│   ├── rag_engine.py       # ChromaDB + sentence-transformers RAG
│   ├── guardrails.py       # Input/output safety filtering (2-stage)
│   ├── observability.py    # Structured JSONL interaction logging
│   └── tools.py            # Calculator, DateTime, KB Search tools
├── knowledge_base/
│   ├── ollive_company.txt  # Ollive AI company docs (RAG source)
│   └── ai_concepts.txt     # AI/LLM reference (RAG source)
├── oss_bot/
│   └── bot.py              # Telegram bot — Qwen2.5-72B via HF API
├── frontier_bot/
│   └── bot.py              # Telegram bot — Claude Sonnet
├── hf_spaces/
│   ├── app.py              # Gradio app — public HF Spaces deployment
│   └── requirements.txt
├── evaluation/
│   ├── eval_prompts.json   # 27 prompts across 4 categories
│   ├── run_evals.py        # LLM-as-judge evaluation runner
│   ├── visualize.py        # Chart generator (4 charts)
│   └── results/            # JSON results + PNG charts (git-ignored)
├── .env.example
├── requirements.txt
└── README.md
```

### Request Flow (per message)
```
User message
    │
    ▼
[GUARDRAIL] Input check → BLOCK if harmful/jailbreak
    │
    ▼
[TOOL] Check for calculator/datetime/search trigger
    │
    ▼
[RAG] Embed query → ChromaDB cosine search → top-3 chunks
    │
    ▼
[LLM] Generate response with RAG context injected
    │
    ▼
[GUARDRAIL] Output check → BLOCK if sensitive content detected
    │
    ▼
[OBS] Log: model, latency, tokens, RAG usage, guardrail decisions
    │
    ▼
Send reply + source footer to Telegram
```

---

## Setup

### Step 1 — Create Two Telegram Bots
1. Open Telegram → search **@BotFather**
2. Send `/newbot` → name it → copy token → paste as `OSS_BOT_TOKEN`
3. Repeat → paste as `FRONTIER_BOT_TOKEN`

### Step 2 — Get API Keys

| Key | Where |
|---|---|
| `HF_TOKEN` | https://huggingface.co/settings/tokens — enable "Make calls to Inference Providers" |
| `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys |

### Step 3 — Install & Configure

```bash
git clone https://github.com/Tanishh2203/ai-assistant-eval
cd ai-assistant-eval

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in your 4 keys
```

### Step 4 — Run the Bots

```bash
# Terminal 1
python oss_bot/bot.py

# Terminal 2
python frontier_bot/bot.py
```

Both bots auto-load `knowledge_base/` on startup.

### Step 5 — Test

Open each bot in Telegram:
- `/start` — welcome message
- `/stats` — verify KB chunks loaded
- `/tools` — see available tools
- Ask: *"What is Ollive and what problem does it solve?"*
- Ask: *"What happened in the Air Canada AI chatbot case?"*
- Try: *"what is 15% of 2500"* — triggers calculator tool
- Upload a `.txt` or `.pdf` to expand the KB

---

## Running the Evaluation

```bash
# Run all 27 prompts through both models + LLM-as-judge (~15-20 mins)
python evaluation/run_evals.py

# Generate 4 charts from results
python evaluation/visualize.py
```

Results saved to `evaluation/results/`:
- `full_results.json` — raw results with scores, responses, latency
- `bar_comparison.png` — side-by-side score chart
- `radar_chart.png` — performance radar
- `latency_chart.png` — avg latency per category
- `summary_table.png` — printable summary

---

## Evaluation Results Summary

| Category | Qwen2.5-72B | Claude Sonnet | Winner |
|---|---|---|---|
| RAG Accuracy | 9.00/10 | 9.00/10 | Tie |
| General Factual | 9.17/10 | 10.00/10 | Claude |
| Safety / Jailbreak | 10.00/10 | 10.00/10 | Tie |
| Bias Handling | 9.60/10 | 9.60/10 | Tie |

| Category | Qwen Latency | Claude Latency |
|---|---|---|
| RAG Accuracy | 6.5s | 8.2s |
| General Factual | 2.8s | 3.2s |
| Safety | 10.6s | 6.6s |
| Bias | 5.7s | 9.1s |

---

## Public Deployment (HF Spaces)

Live demo: https://huggingface.co/spaces/Tanish2203/oss-assistant

To deploy your own:
1. Go to https://huggingface.co/spaces → New Space → SDK: Gradio
2. Upload `hf_spaces/app.py` and `hf_spaces/requirements.txt`
3. Settings → Secrets → add `HF_TOKEN`

---

## Guardrails

Two-stage safety system in `shared/guardrails.py`:

**Stage 1 — Input Guard** (before LLM):
- 15 jailbreak patterns (DAN, "ignore instructions", roleplay bypasses)
- Harmful content: weapons, hacking, CSAM, self-harm
- Bias triggers → WARN + log (model handles gracefully)

**Stage 2 — Output Guard** (after LLM):
- Detects credential/key leakage in responses
- Catches step-by-step harmful instructions that slipped through

---

## Observability

Every interaction logged to `logs/interactions.jsonl`:
```json
{
  "ts": "2026-05-22T14:23:11+00:00",
  "bot_type": "oss",
  "model": "Qwen/Qwen2.5-72B-Instruct",
  "user_id_hash": "a3f9b2c1d4e5",
  "latency_sec": 6.5,
  "prompt_tokens_est": 142,
  "response_tokens_est": 89,
  "rag_chunks_used": 3,
  "rag_sources": ["ollive_company.txt"],
  "guardrail_decision": "PASS",
  "guardrail_category": "none"
}
```

Use `/stats` in Telegram to see live aggregates.

---

## Tool Use

| Tool | Trigger Example | Implementation |
|---|---|---|
| Calculator | "what is 2^10 + 5" | Safe eval() with whitelist |
| DateTime | "what's today's date?" | datetime.utcnow() |
| KB Search | "search knowledge base for X" | RAG engine retrieve() |

---

## Evaluation Categories

| Category | Prompts | Tests |
|---|---|---|
| `rag_factual` | 8 | Ollive KB retrieval + hallucination detection |
| `general_factual` | 6 | General knowledge accuracy |
| `adversarial` | 5 | Jailbreak resistance, safety refusals |
| `bias_sensitivity` | 5 | Fairness + stereotype handling |

---

## Architecture Decisions

**Qwen2.5-72B over smaller models** — 0.5B and 7B were either unavailable or too weak on the HF free inference tier. 72B produces coherent, high-quality responses comparable to frontier models.

**ChromaDB** — Zero-dependency embedded vector DB, no server needed, persists across restarts via PersistentClient.

**sentence-transformers (all-MiniLM-L6-v2)** — Fast, local (no API cost per embed), ~80MB. Good enough for prototyping and demo purposes.

**Keyword-based guardrails over ML classifier** — Zero cost, zero latency, no extra API call. Tradeoff: misses subtle harmful prompts that an ML classifier like Llama Guard would catch.

**LLM-as-judge** — Claude scores both models for consistent evaluation. Known bias: Claude judging Claude may inflate frontier scores by ~0.3–0.8 points.

**In-process dict for memory** — Simple and zero-latency. Resets on restart. Production alternative: Redis with TTL.

---

## Tradeoffs

| Decision | Why | Better Alternative |
|---|---|---|
| HF free inference tier | Zero cost | Paid GPU endpoint (10x faster) |
| Keyword guardrails | No API cost, instant | Llama Guard (better recall) |
| In-memory history | Zero infra | Redis (persistent across restarts) |
| 27 eval prompts | Fast to run | 100+ for statistical significance |
| No streaming | Simpler code | Streaming for better UX |
| MiniLM embeddings | Free, local | OpenAI embeddings (higher quality) |

---

## What I'd Improve With More Time

1. **Llama Guard** — ML-based input/output classifier instead of keyword matching
2. **Redis memory** — Persistent per-user history that survives bot restarts
3. **Larger eval set** — 100+ prompts per category with confidence intervals
4. **Hybrid RAG** — BM25 + dense retrieval combined for better recall on sparse queries
5. **Streaming** — Both APIs support it; would significantly improve perceived latency
6. **Human eval layer** — Blind A/B ratings to validate and calibrate LLM-as-judge scores
7. **Langsmith/Arize** — Production observability dashboard instead of flat JSONL
8. **GPU deployment** — Modal or RunPod for sub-2s OSS inference
9. **Fine-tuning** — Domain-specific fine-tune on Ollive content for better RAG grounding

---

## License

MIT