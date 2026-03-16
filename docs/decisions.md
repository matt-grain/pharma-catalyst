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
