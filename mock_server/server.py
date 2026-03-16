"""
Mock LLM API server — a human-in-the-loop mailbox for testing.

Exposes an OpenAI-compatible /v1/chat/completions endpoint.
Two modes for responding to requests:

  1. **Terminal mode** (default): Displays request in terminal, waits for human input
  2. **Mailbox mode** (--mailbox): Queues requests for pickup via REST API,
     so Claude or any HTTP client can read and respond programmatically

Usage:
    # Terminal mode (human types responses):
    uv run python mock_server/server.py

    # Mailbox mode (API-driven responses):
    uv run python mock_server/server.py --mailbox

    Then in .env:
        LLM_MODEL=openai/mock-model
        OPENAI_API_KEY=mock-key
        OPENAI_API_BASE=http://localhost:8642/v1

Mailbox API:
    GET  /mailbox              — list pending requests
    GET  /mailbox/{request_id} — get a specific pending request (full messages)
    POST /mailbox/{request_id} — submit a response for a pending request
    GET  /canned               — list all canned responses
    GET  /canned/suggest?request_id=N — suggest canned responses for a request

For AG2, the review_panel.py provider detection will use base_url automatically.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from threading import Event, Lock

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .canned_responses import CANNED_RESPONSES, match_canned_responses

app = FastAPI(title="Mock LLM Mailbox")


def _looks_like_json(text: str) -> bool:
    """Check if text contains a JSON object (possibly wrapped in other text)."""
    stripped = text.strip()
    # Direct JSON
    if stripped.startswith("{") and stripped.endswith("}"):
        return True
    # JSON inside markdown code block
    if "```" in stripped and "{" in stripped:
        return True
    # JSON after "Final Answer:" prefix
    if "Final Answer:" in stripped:
        after = stripped.split("Final Answer:")[-1].strip()
        return after.startswith("{")
    return False


def _extract_json(text: str) -> str:
    """Extract the JSON object from text that may contain surrounding prose."""
    import re
    stripped = text.strip()
    # Try: after "Final Answer:"
    if "Final Answer:" in stripped:
        stripped = stripped.split("Final Answer:")[-1].strip()
    # Try: markdown code block
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if md_match:
        return md_match.group(1)
    # Try: first complete JSON object
    brace_start = stripped.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(stripped)):
            if stripped[i] == "{":
                depth += 1
            elif stripped[i] == "}":
                depth -= 1
                if depth == 0:
                    return stripped[brace_start : i + 1]
    return stripped

# Serialize terminal I/O across concurrent requests
_input_lock = Lock()

# Request counter for display
_request_count = 0

# Mailbox mode flag (set via --mailbox CLI arg)
_mailbox_mode = False

# Pending requests waiting for a response (mailbox mode only)
# Maps request_id → {"messages": [...], "model": str, "tools": [...], "event": Event, "response": str | None}
_pending: dict[int, dict] = {}
_pending_lock = Lock()

# Box drawing for pretty terminal output
WIDTH = 100


def _box(title: str, lines: list[str]) -> str:
    header = f" {title} "
    pad = max(0, WIDTH - len(header) - 2)
    left = pad // 2
    right = pad - left
    parts = [f"\n╭{'─' * left}{header}{'─' * right}╮"]
    for line in lines:
        while len(line) > WIDTH - 4:
            parts.append(f"│  {line[:WIDTH - 4]:<{WIDTH - 2}}│")
            line = line[WIDTH - 4:]
        parts.append(f"│  {line:<{WIDTH - 2}}│")
    parts.append(f"╰{'─' * WIDTH}╯")
    return "\n".join(parts)


def _format_messages(messages: list[dict]) -> list[str]:
    """Format chat messages for terminal display."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        name = msg.get("name", "")
        content = msg.get("content", "")
        label = f"[{role}]" if not name else f"[{role}: {name}]"
        # Show first 200 chars of each message
        preview = content[:200].replace("\n", " ")
        if len(content) > 200:
            preview += f"... ({len(content)} chars total)"
        lines.append(f"{label} {preview}")
    return lines


def _terminal_response(req_id: int, messages: list[dict], model: str, tools: list) -> str:
    """Get response via terminal input (original mode)."""
    # Display the request
    header_lines = [
        f"Request #{req_id}  |  Model: {model}  |  Messages: {len(messages)}",
        f"Tools available: {', '.join(t['function']['name'] for t in tools) if tools else 'none'}",
        "",
    ]
    msg_lines = _format_messages(messages)
    print(_box(f"📬 Incoming Request #{req_id}", header_lines + msg_lines))

    # Show suggested canned responses
    suggestions = match_canned_responses(messages)
    if suggestions:
        print(f"\n📋 Suggested canned responses:")
        for i, s in enumerate(suggestions[:5], 1):
            preview = s["response"][:80].replace("\n", " ")
            print(f"   {i}. [{s['id']}] {s['label']} — {preview}...")

    # Prompt for response
    print(f"\n💬 Type your response for request #{req_id}")
    print("   (multi-line: end with an empty line, or 'EOF' on its own line)")
    print("   Shortcuts: 'a' = approve, 'r' = reject, '1'-'5' = use suggested canned response")
    print("─" * WIDTH)

    response_lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "EOF" or (line == "" and response_lines):
            break
        response_lines.append(line)

    raw_response = "\n".join(response_lines).strip()

    # Shortcuts
    if raw_response == "a":
        raw_response = json.dumps({
            "decision": "approved",
            "feedback": "Approved by human reviewer.",
            "confidence": 1.0,
            "concerns": [],
        })
    elif raw_response == "r":
        raw_response = json.dumps({
            "decision": "rejected",
            "feedback": "Rejected by human reviewer.",
            "confidence": 1.0,
            "concerns": ["Human reviewer rejected this proposal"],
        })
    elif raw_response in ("1", "2", "3", "4", "5") and suggestions:
        idx = int(raw_response) - 1
        if idx < len(suggestions):
            raw_response = suggestions[idx]["response"]
            print(f"   → Using canned response: {suggestions[idx]['label']}")

    if not raw_response:
        raw_response = "OK"

    print(_box(f"📤 Response #{req_id} sent", [raw_response[:200]]))
    return raw_response


def _mailbox_response(req_id: int, messages: list[dict], model: str, tools: list) -> str:
    """Queue request and wait for response via mailbox API."""
    event = Event()
    entry = {
        "messages": messages,
        "model": model,
        "tools": [t["function"]["name"] for t in tools] if tools else [],
        "event": event,
        "response": None,
        "timestamp": time.time(),
    }

    with _pending_lock:
        _pending[req_id] = entry

    print(_box(f"📬 Request #{req_id} queued (mailbox)", [
        f"Model: {model}  |  Messages: {len(messages)}",
        f"Waiting for response via POST /mailbox/{req_id}",
    ]))

    # Block until a response is submitted via the API
    event.wait()

    response = entry["response"] or "OK"

    # Clean up
    with _pending_lock:
        _pending.pop(req_id, None)

    print(_box(f"📤 Response #{req_id} delivered", [response[:200]]))
    return response


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    global _request_count
    body = await request.json()

    model = body.get("model", "mock-model")
    messages = body.get("messages", [])
    tools = body.get("tools", [])

    with _input_lock:
        _request_count += 1
        req_id = _request_count

    if _mailbox_mode:
        import asyncio
        raw_response = await asyncio.to_thread(
            _mailbox_response, req_id, messages, model, tools,
        )
    else:
        import asyncio
        raw_response = await asyncio.to_thread(
            _terminal_response, req_id, messages, model, tools,
        )

    # Build OpenAI-compatible response
    completion_id = f"chatcmpl-mock-{uuid.uuid4().hex[:12]}"

    # If tools were in the request AND the response looks like JSON,
    # wrap it as a tool_call (CrewAI/instructor expects this for structured output)
    message: dict
    finish_reason: str
    if tools and _looks_like_json(raw_response):
        tool_name = tools[0]["function"]["name"] if tools else "unknown"
        # Extract just the JSON part if wrapped in text
        json_str = _extract_json(raw_response)
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_mock_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json_str,
                    },
                }
            ],
        }
        finish_reason = "tool_calls"
    else:
        message = {
            "role": "assistant",
            "content": raw_response,
        }
        finish_reason = "stop"

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m.get("content", "")) // 4 for m in messages),
            "completion_tokens": len(raw_response) // 4,
            "total_tokens": 0,
        },
    })


# ── Mailbox API ───────────────────────────────────────────────────────


@app.get("/mailbox")
async def list_pending() -> JSONResponse:
    """List all pending requests waiting for a response."""
    with _pending_lock:
        items = []
        for req_id, entry in _pending.items():
            last_msg = entry["messages"][-1] if entry["messages"] else {}
            items.append({
                "request_id": req_id,
                "model": entry["model"],
                "message_count": len(entry["messages"]),
                "tools": entry["tools"],
                "last_role": last_msg.get("role", ""),
                "last_name": last_msg.get("name", ""),
                "preview": (last_msg.get("content", ""))[:200],
                "age_seconds": round(time.time() - entry["timestamp"], 1),
            })
    return JSONResponse({"pending": items, "count": len(items)})


@app.get("/mailbox/{request_id}")
async def get_pending_request(request_id: int) -> JSONResponse:
    """Get full details of a pending request."""
    with _pending_lock:
        entry = _pending.get(request_id)
        if not entry:
            raise HTTPException(404, f"Request #{request_id} not found or already answered")
        # Suggest canned responses
        suggestions = match_canned_responses(entry["messages"])
        return JSONResponse({
            "request_id": request_id,
            "model": entry["model"],
            "messages": entry["messages"],
            "tools": entry["tools"],
            "age_seconds": round(time.time() - entry["timestamp"], 1),
            "suggested_canned": [
                {"id": s["id"], "label": s["label"], "response_preview": s["response"][:200]}
                for s in suggestions[:5]
            ],
        })


class MailboxResponse(BaseModel):
    response: str | None = None
    canned_id: str | None = None


@app.post("/mailbox/{request_id}")
async def submit_response(request_id: int, body: MailboxResponse) -> JSONResponse:
    """Submit a response for a pending request (either raw text or canned_id)."""
    with _pending_lock:
        entry = _pending.get(request_id)
        if not entry:
            raise HTTPException(404, f"Request #{request_id} not found or already answered")

    # Resolve response text
    if body.canned_id:
        canned = next((c for c in CANNED_RESPONSES if c["id"] == body.canned_id), None)
        if not canned:
            raise HTTPException(400, f"Unknown canned_id: {body.canned_id}")
        response_text = canned["response"]
    elif body.response:
        response_text = body.response
    else:
        raise HTTPException(400, "Provide either 'response' or 'canned_id'")

    # Deliver the response
    entry["response"] = response_text
    entry["event"].set()

    return JSONResponse({
        "status": "delivered",
        "request_id": request_id,
        "response_preview": response_text[:200],
    })


# ── Canned Responses API ─────────────────────────────────────────────


@app.get("/canned")
async def list_canned() -> JSONResponse:
    """List all available canned responses."""
    return JSONResponse({
        "canned": [
            {
                "id": c["id"],
                "label": c["label"],
                "context_hint": c["context_hint"],
                "response_preview": c["response"][:200],
            }
            for c in CANNED_RESPONSES
        ],
        "count": len(CANNED_RESPONSES),
    })


# ── Standard endpoints ───────────────────────────────────────────────


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """Fake models endpoint for litellm compatibility."""
    return JSONResponse({
        "data": [
            {"id": "mock-model", "object": "model", "owned_by": "mock-server"},
        ]
    })


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "mode": "mailbox" if _mailbox_mode else "terminal",
        "requests_served": _request_count,
        "pending": len(_pending),
    }


def main() -> None:
    global _mailbox_mode
    port = 8642

    # Parse CLI args
    args = sys.argv[1:]
    if "--mailbox" in args:
        _mailbox_mode = True
        args.remove("--mailbox")
    if args:
        port = int(args[0])

    mode_label = "MAILBOX" if _mailbox_mode else "TERMINAL"
    mode_info = (
        [
            f"Mode: {mode_label} — requests queued for API pickup",
            "",
            "Mailbox API:",
            "  GET  /mailbox              — list pending requests",
            "  GET  /mailbox/{id}         — get request details + suggested canned",
            "  POST /mailbox/{id}         — submit response (JSON: {response: ...} or {canned_id: ...})",
            "  GET  /canned               — list all canned responses",
        ]
        if _mailbox_mode
        else [
            f"Mode: {mode_label} — you type responses in this terminal",
            "",
            "Requests will appear here. You type the LLM response.",
            "Shortcuts: 'a' = approve, 'r' = reject, '1'-'5' = canned response",
        ]
    )

    print(_box("🧪 Mock LLM Mailbox Server", [
        f"Listening on http://localhost:{port}/v1",
        "",
        *mode_info,
        "",
        "Configure .env:",
        "  LLM_MODEL=openai/mock-model",
        "  OPENAI_API_KEY=mock-key",
        f"  OPENAI_API_BASE=http://localhost:{port}/v1",
    ]))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
