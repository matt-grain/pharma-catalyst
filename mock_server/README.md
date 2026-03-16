# Mock LLM Mailbox Server

Zero-cost testing for the pharma-agents pipeline by intercepting all LLM calls.

## Quick Start

```bash
# Terminal mode (you type responses):
uv run python -m mock_server.server

# Mailbox mode (API-driven responses):
uv run python -m mock_server.server --mailbox
```

Configure `.env`:
```
LLM_MODEL=openai/mock-model
OPENAI_API_KEY=mock-key
OPENAI_API_BASE=http://localhost:8642/v1
```

## Two Response Modes

### Terminal Mode (default)
Requests appear in terminal. You type responses. Shortcuts:
- `a` = approve (review panel JSON)
- `r` = reject (review panel JSON)
- `1`-`5` = use suggested canned response

### Mailbox Mode (`--mailbox`)
Requests queue for pickup via REST API. Use curl, Python, or Claude to respond.

## Mailbox API

| Endpoint | Description |
|----------|-------------|
| `GET /mailbox` | List pending requests (preview, age, tools) |
| `GET /mailbox/{id}` | Full request details + suggested canned responses |
| `POST /mailbox/{id}` | Submit response: `{"response": "..."}` or `{"canned_id": "..."}` |
| `GET /canned` | List all 18 canned responses |
| `GET /health` | Server status |

### Example: Respond via Python (preferred — avoids JSON escaping issues with curl)

```python
import urllib.request, json

payload = json.dumps({"response": "your response"}).encode()
req = urllib.request.Request(
    "http://localhost:8642/mailbox/1",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    print(json.loads(resp.read()))
```

## Mailbox Response Hints

The `GET /mailbox/{id}` response now includes three helper fields:

- **`response_hint`** — Tells you the agent type, expected format, and gives an example response.
  Detects hypothesis, implementation, evaluator, review panel experts, and moderator automatically.
- **`react_tools`** — Extracted from the CrewAI prompt: tool name → params (with types and required flags).
  Use this to get the correct parameter names for `Action Input`.
- **`tool_schemas`** — OpenAI function call schemas (for structured output tools like `HypothesisOutput`).

Example `response_hint` for a hypothesis agent request:
```json
{
  "agent": "hypothesis",
  "format": "ReAct then structured JSON",
  "react_example": "Thought: ...\nAction: search_tooluniverse\nAction Input: {\"query\": \"<string>\"}",
  "final_answer_example": {"proposal": "...", "reasoning": "...", ...}
}
```

**Critical:** Always check `react_tools` for the correct parameter names. CrewAI will
silently fail if you use `{"argument": "..."}` instead of `{"query": "..."}` — it retries
from scratch without any error message, wasting all 3 attempts.

## How to Answer Each Agent

The pipeline makes LLM calls in this order. Each agent type has a different
response format — getting it wrong causes retries and wasted requests.

### 1. Hypothesis Agent (CrewAI)

**How to detect:** `tools` field contains `HypothesisOutput`.

The hypothesis agent operates in **two modes**:

**Mode A — ReAct tool usage (intermediate calls):**
The agent has access to tools like `search_tooluniverse`, `query_literature`,
`discover_skills`, `load_skill`, `lookup_compound`, `fetch_more_papers`, and
`read_train_py`. When it calls a tool via ReAct format, CrewAI executes the tool
locally and returns the result as an Observation. The LLM then gets called again
with the observation in context (message_count > 1).

```
Thought: I should search for validated toxicity prediction approaches.
Action: search_tooluniverse
Action Input: {"query": "toxicity prediction"}
```

**CRITICAL:** Use `react_tools` from the mailbox response to get correct param names.
Common mistakes:
- `search_tooluniverse` takes `{"query": "..."}` NOT `{"argument": "..."}`
- `lookup_compound` takes `{"compound": "..."}` NOT `{"argument": "..."}`
- `discover_skills` takes `{"keyword": "..."}` NOT `{"argument": "..."}`
- `query_literature` takes `{"query": "..."}` NOT `{"argument": "..."}`

**Mode B — Final structured output:**
After using tools (or skipping them), respond with pure JSON matching HypothesisOutput.
The server auto-wraps it as an OpenAI `tool_calls` response (CrewAI/instructor requires this).

```json
{
  "proposal": "Add MACCS keys fingerprints...",
  "reasoning": "MACCS keys capture toxicophore substructures...",
  "change_description": "In train.py, add MACCSkeys.GenMACCSKeys()...",
  "literature_insight": "Multi-fingerprint fusion improves ADMET prediction..."
}
```

**Pitfall:** Do NOT use ReAct format (Thought/Action) here — just the raw JSON.
CrewAI will retry up to 4 times if it can't parse the structured output.

### 2. Review Panel — 5 Panelists + 1 Moderator (AG2)

**How to detect:** No `tools` in request. Messages grow with each speaker.
The `last_name` field shows the current speaker (e.g. `Statistician`).

**Round-robin order:** Statistician → Medicinal_Chemist → Devils_Advocate →
Team_Memory_Analyst → Pharma_Ethics_Reviewer → Moderator

**Panelists (5 agents):** Plain text, 3-5 sentences of expert critique.
```
The proposal is statistically sound. Adding 167 MACCS features
to 1400 ClinTox samples maintains a safe features-to-samples ratio...
```

**Moderator (last agent):** JSON verdict. Must include `decision`, `feedback`,
`confidence`, `concerns`. The review_panel.py parser tries JSON first, falls
back to text extraction, then defaults to "approved" with confidence 0.3.

```json
{
  "decision": "approved",
  "feedback": "Panel finds proposal well-grounded...",
  "confidence": 0.82,
  "concerns": ["Monitor feature correlation"]
}
```

Decisions: `approved` | `revised` (must include `revised_proposal` +
`revised_reasoning`) | `rejected`

### 3. Implementation Agent (CrewAI — ReAct format)

**How to detect:** Message contains "Implement the following APPROVED proposal"
and tools include `read_train_py`, `write_train_py`, `code_check`, `run_train_py`.

**Format:** CrewAI ReAct format with Thought/Action/Action Input.

**Typical sequence (4-8 LLM calls):**

```
# Call 1: Read current code
Thought: I need to read the current train.py.
Action: read_train_py
Action Input: {"argument": "read"}

# Call 2: Write modified code (after seeing Observation with file content)
Thought: I'll implement the proposed changes.
Action: write_train_py
Action Input: {"content": "full python code here"}

# Call 3: Check for errors
Thought: Let me verify the code.
Action: code_check
Action Input: {"argument": "check"}

# Call 4: If errors, fix and rewrite. If OK, finish:
Thought: I now know the final answer
Final Answer: I have implemented the changes: [summary]
```

**Critical pitfall — f-string escaping:** When sending Python code containing
f-strings through JSON, curly braces need escaping. Build the code string in
Python and use `json.dumps()` to handle escaping automatically:

```python
code = 'print(f"Score: {score:.4f}")\n'
response = 'Action: write_train_py\nAction Input: {"content": ' + json.dumps(code) + '}'
```

NEVER manually escape f-strings in JSON — it will break.

### 4. Evaluator Agent (CrewAI — ReAct format)

**How to detect:** Message contains "Run the modified train.py and evaluate".
Tools include `read_train_py` and `run_train_py` (NO write access).

**Format:** Same ReAct format as implementation agent.

```
# Call 1: Run training
Thought: I need to run the training script.
Action: run_train_py
Action Input: {"argument": "run"}

# Call 2: Report results (after seeing Observation with score)
Thought: I now know the final answer
Final Answer: ROC_AUC: 0.9234
BASELINE: 0.6915
IMPROVEMENT: +33.5%
RECOMMENDATION: KEEP
```

**Critical pitfall — anti-loop:** CrewAI blocks repeated actions with identical
input. If `run_train_py` fails (e.g. syntax error in train.py), the evaluator
CANNOT retry it with the same `{"argument": "run"}` input. It will get:
"I tried reusing the same input, I must stop using this action input."
In this case, give a Final Answer directly with the error details.

## Auto Tool-Call Wrapping

The server auto-detects the response format:
- **Request has `tools` + response is JSON** → wrapped as OpenAI `tool_calls`
  (required by CrewAI/instructor for structured output parsing)
- **No tools or plain text response** → returned as `content`
  (used by AG2 review panel and CrewAI ReAct agents)

This is automatic — just send the response text and the server handles the rest.

## Canned Responses

18 pre-built responses in `canned_responses.py`. Use by ID:

| Category | IDs |
|----------|-----|
| Hypothesis | `hypothesis_1` (RDKit descriptors), `hypothesis_2` (Random Forest) |
| Statistician | `statistician_approve`, `statistician_concern` |
| Chemist | `chemist_approve` |
| Devil's Advocate | `devil_mild`, `devil_harsh` |
| Memory Analyst | `memory_novel`, `memory_repeat` |
| Ethics | `ethics_ok` |
| Moderator | `moderator_approve`, `moderator_revise`, `moderator_reject` |
| Implementation | `impl_read_train`, `impl_write_code`, `impl_run_train` |
| Evaluator | `eval_improved`, `eval_no_change` |

## Typical Full Run

A complete iteration produces ~19 requests:

| Requests | Agent | Framework | Response Format |
|----------|-------|-----------|-----------------|
| 1 | Hypothesis | CrewAI | JSON → auto tool_call |
| 2-7 | Review panel (6 agents) | AG2 | Plain text (Moderator: JSON) |
| 8-15 | Implementation | CrewAI | ReAct (Thought/Action) |
| 16-19 | Evaluator | CrewAI | ReAct (Thought/Action) |

## New Tools (added 2026-03-16)

These tools call real external APIs — they work even with the mock server.
The LLM doesn't call them directly; CrewAI executes them locally based on ReAct actions.

| Tool | Agent | API | Parameter |
|------|-------|-----|-----------|
| `search_tooluniverse` | Hypothesis | aiscientist.tools catalog (1,936 tools) | `{"query": "toxicity prediction"}` |
| `search_pubmed` | Archivist | NCBI E-utilities (PubMed) | `{"query": "ClinTox toxicity ML"}` |
| `lookup_compound` | Hypothesis | PubChem PUG REST | `{"compound": "aspirin"}` or `{"compound": "CCO"}` |
| `validate_experimental` | Evaluator | PubChem PUG REST | `{"smiles_list": ["CCO"], "property": "logP"}` |
| `discover_skills` | Hypothesis | Local skill files | `{"keyword": "drug"}` or `{"keyword": "all"}` |

**Note:** When using the mock server, these tools produce real results (network calls)
while the LLM reasoning is mocked. This means the hypothesis agent will see actual
ToolUniverse search results, PubMed papers, and PubChem compound data in its context.

## Known Issues (from live testing 2026-03-15)

### 1. F-string double-encoding
Writing Python f-strings through JSON tool calls mangles `\n` and `{}`.
A real LLM hits this too. **Mitigation:** Always build code strings in Python
and use `json.dumps()`. Consider adding a `edit_train_py` patch tool to avoid
full-file rewrites.

### 2. CrewAI anti-loop blocks valid retries
If `run_train_py` fails due to a code bug (fixed by implementation agent),
the evaluator can't retry because CrewAI sees the same action+input.
**Mitigation:** The `cache_function` override on CodeCheckTool already handles
this for code_check. RunTrainPyTool needs the same treatment, or accept an
optional `reason` parameter so the input differs on retry.

### 3. CrewAI mega-prompt
CrewAI combines system prompt + agent backstory + tools + task description +
output schema + format instructions into ONE large user message (~5000 tokens).
This is a CrewAI/litellm design choice, not a bug. **Mitigation:** Keep task
descriptions concise; long context in `{agent_memory}` or `{experiment_history}`
should be summarized before injection.

### 4. write_train_py requires FULL file content
The LLM must reproduce the entire file including unchanged sections. Small
models may introduce subtle mutations. **Mitigation:** Add an `edit_train_py`
tool that accepts line-range patches instead of full rewrites.

### 5. Evaluator has no diff visibility
The evaluator can read and run train.py but can't see what changed from
baseline. It reports the score but can't explain the delta. Minor issue —
the hypothesis context already describes the change.

### 6. Wrong parameter names cause silent retries (2026-03-16)
CrewAI tools have specific parameter names (e.g., `query`, `compound`, `keyword`).
If the mock responder sends the wrong param name (e.g., `{"argument": "toxicity"}`
instead of `{"query": "toxicity"}`), CrewAI silently fails the tool call and
retries the entire task from scratch — no error message, just a fresh request
with `message_count=1`. After `max_iter` retries, the task fails.
**Mitigation:** Always check the `react_tools` field in `GET /mailbox/{id}` —
it shows the exact parameter names and types extracted from the CrewAI prompt.
The `response_hint` field also provides a ready-to-use example.
