# Architecture Decision Records

## 2026-03-15 — AutoGen for Adversarial Review Panel

**Status:** accepted
**Context:** Hypotheses go straight into implementation without validation. A bad hypothesis wastes 3-5 minutes of model training before the evaluator discovers it failed. We need a domain-expert review step between hypothesis and implementation.
**Decision:** Use AutoGen (AG2) GroupChat for the review panel instead of adding more CrewAI agents. Three specialists (Statistician, Medicinal Chemist, Devil's Advocate) debate the proposal in a round-robin GroupChat, then a Moderator issues a structured verdict.
**Alternatives considered:**
- Adding review agents to the existing CrewAI crew — rejected because CrewAI's sequential task model doesn't support multi-turn debate where agents respond to each other's arguments.
- A single "reviewer" agent — rejected because adversarial multi-perspective review catches more issues than a single viewpoint.
- A custom debate loop without a framework — rejected as unnecessarily reinventing what GroupChat already provides.
**Consequences:**
- Two agent framework dependencies (CrewAI + AG2), increasing dependency surface.
- Each framework is used for its strength: CrewAI for sequential pipelines, AutoGen for adversarial debate.
- ~30-60s added latency per review, offset by avoiding 3-5 min wasted training on bad hypotheses.
- Fallback to legacy single-crew flow available via `--no-review` flag.

## 2026-03-16 — Scientific Data Integration (PubMed, PubChem) + ToolUniverse Skills

**Status:** accepted
**Context:** The demo only searches arxiv for literature and has no way to validate predictions
against real experimental data. PubMed provides peer-reviewed literature, PubChem provides
experimental molecular properties, and ToolUniverse (Harvard Zitnik Lab) provides curated
workflow playbooks for pharma research.
**Decision:** Wrap PubMed (NCBI E-utilities) and PubChem (PUG REST) as direct HTTP-based CrewAI
BaseTool subclasses — no SDK dependency. Copy 15 ToolUniverse skill playbooks to `.claude/skills/`
for agent workflow guidance via SkillLoaderTool.
**Alternatives considered:**
- ToolUniverse Python SDK — rejected because it eagerly imports heavy ML deps (torch, chemprop,
  admet-ai) causing >60s init time and broken imports on some platforms. Direct REST is zero-dep.
- Semantic Scholar API — considered but deferred; PubMed + arxiv already gives good coverage.
- MCP server for agent access — rejected for autonomous agents (MCP is for interactive Claude Code
  use). BaseTool wrappers are the correct CrewAI integration pattern.
**Consequences:**
- No new Python dependencies — tools use stdlib urllib only.
- Optional `NCBI_API_KEY` env var boosts PubMed rate limit from 3 to 10 req/s.
- 15 ToolUniverse skill directories added to `.claude/skills/` (~500KB total).
- Demo gains: multi-source literature (arxiv + PubMed), experimental data lookups, validation.
