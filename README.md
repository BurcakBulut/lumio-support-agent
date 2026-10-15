# Lumio Support Agent

An AI support agent for Lumio, a fictional SaaS company (Free, Plus and Business plans plus the Lumio Hub device). For every ticket it decides whether to **resolve**, **ask for clarification**, **escalate** to a human or **decline**. It cites knowledge-base sources for every policy statement and enforces refund and identity rules in code.

## Repository contents

| Path | What it is |
|---|---|
| `01_eda.ipynb` | Exploratory data analysis: every file profiled, findings ending in "so the agent will …" |
| `02_agent.ipynb` | The agent: tools, rules, model steps, demo, evaluation and error analysis |
| `data/` | The scenario data, unchanged |
| `results/` | Evaluation results: `eval_results.csv` and `metrics.json` (v2), `*_v1` files (first version) |
| `BRIEF.md` | The project brief |

## How the agent works

1. **Rules, no model:** instruction-like text, escalation keywords, unknown or someone else's order or email, suspended account
2. **Model classifies:** a local `llama3.2:3b` (Ollama) returns a category and 7 request flags as JSON
3. **Code verifies and acts:** flags must match the customer's own words; refund checks (owner, status, window, €100 limit) run inside the refund tool; password resets go only to the registered email
4. **Model retrieves and answers:** picks up to 2 of 28 trusted documents; a contradicting document means escalation; answers only from the documents, with citations

Every ticket gets a decision, reason, tools called, sources, reply, handoff summary (for escalations), model calls, tokens and seconds. Tools are mocked in plain Python.

## How to run

Requirements: [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/).

```bash
git clone https://github.com/BurcakBulut/lumio-support-agent.git
cd lumio-support-agent
uv sync
ollama pull llama3.2:3b
```

Open `02_agent.ipynb` in VS Code, select the project's `.venv` as the kernel and run all cells. The evaluation (sections 7 and 8) runs all 31 test tickets and writes to `results/`. "Today" is fixed at 2026-10-01 (from `data/policy_config.json`). The first model call takes longer while the model loads.

## Results (31 test tickets)

| Metric | v1 | v2 |
|---|---|---|
| Action accuracy | 0.84 | 0.90 |
| Tool correctness | 0.77 | 0.87 |
| Escalation precision / recall | 0.85 / 0.85 | 0.92 / 0.92 |
| Source recall / precision | 0.76 / 0.75 | 0.90 / 0.83 |
| Automation rate (best possible 0.42) | 0.39 | 0.42 |
| Wrong actions with side effects | 2 | 0 |
| Cost per ticket | 1.29 model calls, 626 tokens, 0.91 s | 1.39 model calls, 665 tokens, 0.93 s |

No API cost: the model runs locally. 6 of 31 tickets are decided by rules alone. A manual review found problems in 4 of the 7 answers the model wrote itself (see section 8 of `02_agent.ipynb`).

## Key design decisions

- **Trusted documents only:** highest version per title, no deprecated or partner documents; contradicting documents lead to escalation
- **The model proposes, the code verifies:** no action with side effects unless the customer's own words ask for it
- **Hard limits in code:** the €100 refund limit, refund windows, ownership and registered-email checks live inside the tools
- **Text is data:** ticket and document text is never treated as an instruction; instruction-like text is escalated

## Limitations and next steps

- Small test set (31 tickets); the v2 fix was designed and tested on the same set
- The 3B model is weak at categories (0.56) and free-text answers; answers should be reviewed by a person before sending
- Next: fix the contradicting documents at the source, embedding-based retrieval, a stronger model, shadow mode on real tickets, a real identity check