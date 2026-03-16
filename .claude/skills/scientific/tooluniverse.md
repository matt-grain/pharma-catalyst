# ToolUniverse Integration

Overview of ToolUniverse tools available to pharma-agents via CrewAI BaseTool wrappers.

## Available Agent Tools

These are CrewAI BaseTool wrappers that agents can call directly:

| Agent Tool | ToolUniverse APIs Used | Assigned To |
|------------|----------------------|-------------|
| `search_pubmed` | PubMed_search_articles, SemanticScholar_search_papers | Archivist |
| `lookup_compound` | PubChem_get_CID_by_SMILES, PubChem_get_compound_properties_by_CID, ChEMBL_search_molecules | Hypothesis |
| `predict_admet` | ADMETAI_predict_* (9 endpoints) | Hypothesis |
| `validate_experimental` | PubChem + ChEMBL lookups for experimental values | Evaluator |

## Skill Playbooks

Directory-based skills in `.claude/skills/tooluniverse-*/SKILL.md` provide multi-step
workflow guides. Agents load these via `load_skill("tooluniverse-drug-research")` to get
a structured research playbook with tool names and calling patterns.

Key playbooks for pharma-agents:
- **tooluniverse-drug-research** — Full drug profiling (identity, pharmacology, ADMET, clinical)
- **tooluniverse-literature-deep-research** — Systematic literature review with evidence grading
- **tooluniverse-chemical-compound-retrieval** — Compound disambiguation and cross-DB lookup
- **tooluniverse-chemical-safety** — ADMET + toxicity assessment workflow
- **tooluniverse-drug-repurposing** — Existing drug repurposing analysis
- **tooluniverse-target-research** — Target validation and druggability

## SDK Quick Reference

```python
from tooluniverse import ToolUniverse
tu = ToolUniverse()
tu.load_tools()  # ~1,996 tools

# Call tools directly
tu.tools.PubMed_search_articles(query="...", limit=5)
tu.tools.PubChem_get_compound_properties_by_CID(cid=702)
tu.tools.ADMETAI_predict_toxicity(smiles=["CCO"])  # needs tooluniverse[ml]
tu.tools.ChEMBL_search_molecules(query="aspirin", max_results=3)
tu.tools.SemanticScholar_search_papers(query="...", limit=5)

# Discover tools
tu.find_tools_by_pattern("PubMed")  # returns list[dict]
```
