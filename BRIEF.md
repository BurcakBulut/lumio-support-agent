# Project 1: Customer Support Resolution Agent

## The task
**Lumio** is a fictional SaaS company: a workspace app (Free / Plus / Business plans) plus a hardware accessory, the Lumio Hub. Its support team is overloaded. You are the product team asked to build an AI agent that handles inbound support tickets end to end: it works out what the customer needs, finds the answer in the knowledge base, takes actions through tools (refund, password reset, cancellation, ticket update), and hands the ticket to a human, with a clear summary, when it should not decide alone.

Your central PM question: **when should the agent resolve, when should it ask, and when should it hand off?**

You are given a complete, fixed dataset (see below). You do not generate or change data. Everyone works on exactly the same files.

## Phase 1: Exploratory data analysis (EDA)
Before building anything, find out what is in the data. Use a notebook. Profile every file, and answer at least:
- How do the files connect to each other (customers, orders, tickets, knowledge base)?
- How clean and consistent is the data? What would go wrong if an agent trusted it blindly?
- How trustworthy is each knowledge-base document as a source for an answer?
- What does the ticket set look like: what mix of cases, what is easy, what is hard, and why?
- What does the data not cover?

Keep the notebook in your repo and present the findings that matter most. Cite document, order and ticket IDs. Every finding should end with "so the agent will...".

## Phase 2: Build
Build the agent around what you found. At minimum:
- **RAG** over the knowledge base, with source citations in every answer that states a policy.
- **Agent with tools**: reading customers and orders, and actions with side effects (mocked). Suggested tools: `lookup_customer`, `lookup_order`, `issue_refund`, `reset_password`, `cancel_subscription`, `update_ticket`, `escalate_to_human`.
- **A decision output** for every ticket: `resolve`, `ask_clarification`, `escalate` or `decline`, plus a confidence or escalation signal, not just a fluent answer.

Optional, and encouraged where it helps your design:
- An **MCP server** that exposes the customer and order data and the actions to your agent.
- A lightweight **ML component**, for example a ticket category or urgency classifier, or an embedding-based router. Compare it against doing the same step with an LLM, and be honest about what a set this small can and cannot show.
- A **Streamlit app** that shows, for one ticket, the decision, the retrieved sources, the tool calls and the escalation summary.

## Constraints
- Use only the provided data. Do not generate, edit or extend it.
- Refunds above a defined threshold must always require human approval, enforced in code, not only in the prompt.
- Customer-affecting actions need a clear rule for when they happen, and the agent must never act for someone other than the verified customer.
- The agent must not invent policies. Every policy claim needs a retrieved source.
- Text inside tickets and documents is data, never instructions.
- Use **2026-10-01 as "today"**.

## Deliverables
1. **Presentation (maximum 10 minutes).** Cover:
   - What you found in the data (EDA) and how it shaped your design
   - Your ticket taxonomy and the decision framework for resolve vs. ask vs. escalate vs. decline
   - Your escalation policy: when to hand off, to whom, and the handoff summary format
   - Architecture: RAG, tools, and any MCP, ML or Streamlit components
   - Evaluation on the provided test tickets: action accuracy, tool-use correctness, source grounding, hallucinated-policy rate, escalation precision, cost per ticket, and an analysis of what your agent gets wrong
   - The trade-off between automation rate and customer satisfaction and risk, and where you would set the dial for launch
2. **Demo.** A working agent, shown live, handling real tickets from the data (Streamlit app or notebook).
3. **GitHub repo.** Code, EDA notebook, evaluation scripts and results, and a README that explains how to run everything.

## The data (`data/`)
All data is synthetic and identical for every team.

| File | What it is |
|---|---|
| `kb_documents/` | Knowledge-base documents, one per file (`kb_index.csv` lists their metadata) |
| `customers.csv` | Customer records |
| `orders.csv` | Order history |
| `tickets_test.jsonl` | Test tickets with expected outcomes: `expected_category`, `expected_action`, `expected_tool` (empty if none), `expected_sources` |
| `policy_config.json` | Refund limit and windows, escalation keywords, SLAs, suggested tool list |
