# ToolUniverse Integration — Implementation Plan

## Overview

Add ToolUniverse (Harvard Zitnik Lab, 1,996 scientific tools) to pharma-agents via its Python SDK.
This gives agents PubMed/Semantic Scholar literature search, PubChem/ChEMBL compound lookups,
ADMET-AI predictions, and experimental data validation — all as CrewAI BaseTool wrappers.

**SDK API reference** (verified by testing):
```python
from tooluniverse import ToolUniverse
tu = ToolUniverse()
tu.load_tools()  # loads 1,996 tools

# Call tools via namespace:
tu.tools.PubMed_search_articles(query="...", limit=3)  # returns list[dict]
tu.tools.PubChem_get_compound_properties_by_CID(cid=702)  # returns dict
tu.tools.ADMETAI_predict_physicochemical_properties(smiles=["CCO"])  # returns dict (needs tooluniverse[ml])
tu.tools.ChEMBL_search_molecules(query="aspirin")  # returns dict
tu.tools.SemanticScholar_search_papers(query="...", limit=3)  # returns list[dict]

# Discovery:
tu.find_tools_by_pattern("PubMed")  # returns list[dict] with name, description, parameter keys
```

---

## Task 1: Add `tooluniverse` dependency

**File:** `pyproject.toml`
**Action:** MODIFY — add 1 line to `dependencies`

Add `"tooluniverse>=1.0.0"` to the dependencies list (after `"catboost>=1.2.10"`).

Do NOT add `tooluniverse[ml]` — the `[ml]` extra pulls in heavy GPU deps. ADMET-AI will
gracefully return an error message if `admet-ai` is not installed, and our tool handles that.

**Verification:** `uv sync` completes without errors.

---

## Task 2: Update `.env.example`

**File:** `.env.example`
**Action:** MODIFY — add 2 lines at the end

```
# NCBI API Key (optional, boosts PubMed from 3 to 10 req/s)
# Get free key at: https://www.ncbi.nlm.nih.gov/account/settings/
NCBI_API_KEY=
```

---

## Task 3: Create `src/pharma_agents/tools/tooluniverse.py`

**File:** `src/pharma_agents/tools/tooluniverse.py`
**Action:** NEW (~220 lines)

Create 4 BaseTool subclasses following the exact same pattern as `arxiv.py`:
- Class-level `ClassVar` for rate limiting (`_last_call`, `_calls_done`)
- `@classmethod reset_counters()`
- `_run()` method with rate limiting and max call enforcement
- Lazy init of ToolUniverse instance (shared across tools via module-level singleton)

### Module-level singleton

```python
"""ToolUniverse-based tools for literature, compound, and ADMET lookups."""

import json
import time
from typing import ClassVar

from crewai.tools import BaseTool
from loguru import logger

# Lazy singleton — ToolUniverse loads 1,996 tools on init (~2s)
_tu_instance = None

def _get_tu():
    """Get or create the shared ToolUniverse instance."""
    global _tu_instance
    if _tu_instance is None:
        from tooluniverse import ToolUniverse
        _tu_instance = ToolUniverse()
        _tu_instance.load_tools()
        logger.info(f"ToolUniverse loaded {len(_tu_instance.all_tools)} tools")
    return _tu_instance
```

### Tool 1: `PubMedSearchTool`

```python
class PubMedSearchTool(BaseTool):
    """Search PubMed and Semantic Scholar for peer-reviewed papers."""

    name: str = "search_pubmed"
    description: str = (
        "Searches PubMed and Semantic Scholar for peer-reviewed papers on a topic. "
        "Input: search query (e.g., 'BBBP prediction graph neural network'). "
        "Returns PMID + title + abstract for each result. "
        "Use this alongside search_and_store (which searches arxiv) for comprehensive coverage."
    )
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

    max_calls_per_run: int = 3
    min_interval_seconds: float = 0.34  # 3 req/s for NCBI without API key
    _calls_done: ClassVar[int] = 0
    _last_call: ClassVar[float] = 0.0

    @classmethod
    def reset_counters(cls) -> None:
        cls._calls_done = 0
        cls._last_call = 0.0

    def _run(self, query: str) -> str:
        if PubMedSearchTool._calls_done >= self.max_calls_per_run:
            return f"Max PubMed searches ({self.max_calls_per_run}) reached."

        # Rate limiting
        elapsed = time.time() - PubMedSearchTool._last_call
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        PubMedSearchTool._last_call = time.time()
        PubMedSearchTool._calls_done += 1

        tu = _get_tu()
        results = []

        # 1. PubMed search
        try:
            pubmed_results = tu.tools.PubMed_search_articles(
                query=query, limit=3, include_abstract=True
            )
            # pubmed_results is a list[dict] with keys: pmid, title, authors, journal, pub_date, doi, url
            if isinstance(pubmed_results, list):
                for r in pubmed_results:
                    results.append({
                        "id": f"pmid:{r.get('pmid', '')}",
                        "title": r.get("title", ""),
                        "abstract": r.get("abstract", r.get("title", "")),
                        "date": r.get("pub_date", ""),
                        "source": "pubmed",
                        "url": r.get("url", ""),
                    })
        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")

        # 2. Semantic Scholar search (complementary)
        try:
            ss_results = tu.tools.SemanticScholar_search_papers(
                query=query, limit=2, include_abstract=True
            )
            if isinstance(ss_results, list):
                for r in ss_results:
                    paper_id = r.get("paperId", r.get("paper_id", ""))
                    results.append({
                        "id": f"s2:{paper_id[:12]}",
                        "title": r.get("title", ""),
                        "abstract": r.get("abstract", ""),
                        "date": str(r.get("year", "")),
                        "source": "semantic_scholar",
                    })
            elif isinstance(ss_results, dict) and "data" in ss_results:
                for r in ss_results["data"][:2]:
                    paper_id = r.get("paperId", "")
                    results.append({
                        "id": f"s2:{paper_id[:12]}",
                        "title": r.get("title", ""),
                        "abstract": r.get("abstract", ""),
                        "date": str(r.get("year", "")),
                        "source": "semantic_scholar",
                    })
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")

        if not results:
            return f"No papers found for query: {query}"

        formatted = []
        for r in results:
            abstract = r["abstract"][:300] + "..." if len(r.get("abstract", "")) > 300 else r.get("abstract", "")
            formatted.append(
                f"- **{r['id']}** ({r['date']}, {r['source']}): {r['title']}\n  Abstract: {abstract}\n"
            )

        return f"Found {len(results)} papers for '{query}':\n\n" + "\n".join(formatted)
```

### Tool 2: `CompoundLookupTool`

```python
class CompoundLookupTool(BaseTool):
    """Look up compound properties from PubChem and ChEMBL."""

    name: str = "lookup_compound"
    description: str = (
        "Looks up real experimental properties for a molecule from PubChem and ChEMBL. "
        "Input: SMILES string (e.g., 'CCO' for ethanol) or compound name (e.g., 'aspirin'). "
        "Returns molecular weight, logP, TPSA, and known bioactivities. "
        "Use this to ground hypotheses in real chemistry data."
    )
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

    max_calls_per_run: int = 5
    min_interval_seconds: float = 0.5
    _calls_done: ClassVar[int] = 0
    _last_call: ClassVar[float] = 0.0

    @classmethod
    def reset_counters(cls) -> None:
        cls._calls_done = 0
        cls._last_call = 0.0

    def _run(self, compound: str) -> str:
        if CompoundLookupTool._calls_done >= self.max_calls_per_run:
            return f"Max compound lookups ({self.max_calls_per_run}) reached."

        elapsed = time.time() - CompoundLookupTool._last_call
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        CompoundLookupTool._last_call = time.time()
        CompoundLookupTool._calls_done += 1

        tu = _get_tu()
        compound = compound.strip()
        output_parts = []

        # Resolve CID — try SMILES first, then name
        cid = None
        try:
            # Heuristic: if it contains atoms-like chars and no spaces, treat as SMILES
            if any(c in compound for c in "()=#[]") or (len(compound) < 100 and " " not in compound and compound[0].isupper()):
                result = tu.tools.PubChem_get_CID_by_SMILES(smiles=compound)
                if isinstance(result, dict) and "IdentifierList" in result:
                    cids = result["IdentifierList"].get("CID", [])
                    if cids:
                        cid = cids[0]
            if cid is None:
                result = tu.tools.PubChem_get_CID_by_compound_name(compound_name=compound)
                if isinstance(result, dict) and "IdentifierList" in result:
                    cids = result["IdentifierList"].get("CID", [])
                    if cids:
                        cid = cids[0]
        except Exception as e:
            logger.warning(f"CID lookup failed for '{compound}': {e}")

        if cid is None:
            return f"Could not find compound '{compound}' in PubChem."

        # Get PubChem properties
        try:
            props = tu.tools.PubChem_get_compound_properties_by_CID(cid=cid)
            if isinstance(props, dict) and "PropertyTable" in props:
                prop_list = props["PropertyTable"].get("Properties", [{}])
                if prop_list:
                    p = prop_list[0]
                    output_parts.append(
                        f"**PubChem (CID {cid}):**\n"
                        f"  SMILES: {p.get('CanonicalSMILES', p.get('ConnectivitySMILES', 'N/A'))}\n"
                        f"  MW: {p.get('MolecularWeight', 'N/A')}\n"
                        f"  LogP (XLogP3): {p.get('XLogP', 'N/A')}\n"
                        f"  TPSA: {p.get('TPSA', 'N/A')}\n"
                        f"  HBD: {p.get('HBondDonorCount', 'N/A')}, HBA: {p.get('HBondAcceptorCount', 'N/A')}\n"
                        f"  Rotatable bonds: {p.get('RotatableBondCount', 'N/A')}"
                    )
        except Exception as e:
            logger.warning(f"PubChem properties failed for CID {cid}: {e}")
            output_parts.append(f"PubChem properties lookup failed: {e}")

        # Try ChEMBL cross-reference
        try:
            chembl_results = tu.tools.ChEMBL_search_molecules(query=compound, max_results=1)
            # Handle various response formats
            molecules = []
            if isinstance(chembl_results, dict):
                molecules = chembl_results.get("molecules", chembl_results.get("data", []))
            elif isinstance(chembl_results, list):
                molecules = chembl_results
            if molecules and isinstance(molecules[0], dict):
                mol = molecules[0]
                chembl_id = mol.get("molecule_chembl_id", "unknown")
                output_parts.append(
                    f"\n**ChEMBL ({chembl_id}):**\n"
                    f"  Name: {mol.get('pref_name', 'N/A')}\n"
                    f"  Max phase: {mol.get('max_phase', 'N/A')}\n"
                    f"  Molecule type: {mol.get('molecule_type', 'N/A')}"
                )
        except Exception as e:
            logger.debug(f"ChEMBL lookup skipped: {e}")

        if not output_parts:
            return f"Found CID {cid} but could not retrieve properties."

        return f"Compound lookup for '{compound}':\n\n" + "\n".join(output_parts)
```

### Tool 3: `ADMETPredictTool`

```python
class ADMETPredictTool(BaseTool):
    """Predict ADMET properties using ADMET-AI."""

    name: str = "predict_admet"
    description: str = (
        "Predicts ADMET properties for molecules using ADMET-AI. "
        "Input: JSON with 'smiles' (single SMILES string or comma-separated list) "
        "and optional 'properties' (list: 'physicochemical', 'bbb', 'toxicity', 'solubility', 'cyp', 'bioavailability'). "
        "Default: predicts physicochemical + BBB + toxicity. "
        "Returns predicted values with DrugBank percentiles."
    )
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

    max_calls_per_run: int = 5
    _calls_done: ClassVar[int] = 0

    @classmethod
    def reset_counters(cls) -> None:
        cls._calls_done = 0

    # Map property names to ADMET-AI tool method names
    _PROPERTY_MAP: ClassVar[dict[str, str]] = {
        "physicochemical": "ADMETAI_predict_physicochemical_properties",
        "bbb": "ADMETAI_predict_BBB_penetrance",
        "toxicity": "ADMETAI_predict_toxicity",
        "solubility": "ADMETAI_predict_solubility_lipophilicity_hydration",
        "cyp": "ADMETAI_predict_CYP_interactions",
        "bioavailability": "ADMETAI_predict_bioavailability",
        "clearance": "ADMETAI_predict_clearance_distribution",
    }

    def _run(self, input_data: str) -> str:
        if ADMETPredictTool._calls_done >= self.max_calls_per_run:
            return f"Max ADMET predictions ({self.max_calls_per_run}) reached."
        ADMETPredictTool._calls_done += 1

        # Parse input
        try:
            data = json.loads(input_data)
            smiles_input = data.get("smiles", input_data)
            properties = data.get("properties", ["physicochemical", "bbb", "toxicity"])
        except (json.JSONDecodeError, AttributeError):
            smiles_input = input_data.strip()
            properties = ["physicochemical", "bbb", "toxicity"]

        # Normalize SMILES to list
        if isinstance(smiles_input, str):
            smiles_list = [s.strip() for s in smiles_input.split(",") if s.strip()]
        else:
            smiles_list = list(smiles_input)

        if not smiles_list:
            return "Error: No SMILES provided."
        if len(smiles_list) > 10:
            smiles_list = smiles_list[:10]  # Cap at 10

        tu = _get_tu()
        output_parts = [f"ADMET-AI predictions for {len(smiles_list)} molecule(s):"]

        for prop in properties:
            tool_name = self._PROPERTY_MAP.get(prop)
            if not tool_name:
                output_parts.append(f"\n**{prop}**: Unknown property. Available: {list(self._PROPERTY_MAP.keys())}")
                continue

            try:
                tool_fn = getattr(tu.tools, tool_name)
                result = tool_fn(smiles=smiles_list)

                if isinstance(result, dict) and "error" in result:
                    output_parts.append(f"\n**{prop}**: {result['error']}")
                else:
                    output_parts.append(f"\n**{prop}**: {json.dumps(result, indent=2, default=str)[:1000]}")
            except Exception as e:
                output_parts.append(f"\n**{prop}**: Prediction failed — {e}")

        return "\n".join(output_parts)
```

### Tool 4: `ExperimentalValidationTool`

```python
class ExperimentalValidationTool(BaseTool):
    """Validate predictions against experimental data from ChEMBL/PubChem."""

    name: str = "validate_experimental"
    description: str = (
        "Looks up experimental values for molecules to validate model predictions. "
        "Input: JSON with 'smiles_list' (up to 10 SMILES) and 'property' (e.g., 'solubility', 'logP', 'bbb'). "
        "Returns a comparison table: SMILES | Experimental Value | Source. "
        "Use this to ground ML predictions in real-world data."
    )
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

    max_calls_per_run: int = 3
    _calls_done: ClassVar[int] = 0

    @classmethod
    def reset_counters(cls) -> None:
        cls._calls_done = 0

    def _run(self, input_data: str) -> str:
        if ExperimentalValidationTool._calls_done >= self.max_calls_per_run:
            return f"Max validation calls ({self.max_calls_per_run}) reached."
        ExperimentalValidationTool._calls_done += 1

        try:
            data = json.loads(input_data)
        except json.JSONDecodeError:
            return "Error: Input must be JSON with 'smiles_list' and 'property'."

        smiles_list = data.get("smiles_list", [])
        prop = data.get("property", "logP")

        if not smiles_list:
            return "Error: 'smiles_list' is empty."
        if len(smiles_list) > 10:
            smiles_list = smiles_list[:10]

        tu = _get_tu()
        rows = []

        for smiles in smiles_list:
            row = {"smiles": smiles, "value": "N/A", "source": "none"}

            # Try PubChem first
            try:
                cid_result = tu.tools.PubChem_get_CID_by_SMILES(smiles=smiles)
                cid = None
                if isinstance(cid_result, dict) and "IdentifierList" in cid_result:
                    cids = cid_result["IdentifierList"].get("CID", [])
                    if cids:
                        cid = cids[0]

                if cid:
                    props = tu.tools.PubChem_get_compound_properties_by_CID(cid=cid)
                    if isinstance(props, dict) and "PropertyTable" in props:
                        p = props["PropertyTable"].get("Properties", [{}])[0]
                        # Map requested property to PubChem field
                        prop_map = {
                            "logP": "XLogP",
                            "logp": "XLogP",
                            "mw": "MolecularWeight",
                            "molecular_weight": "MolecularWeight",
                            "tpsa": "TPSA",
                            "hbd": "HBondDonorCount",
                            "hba": "HBondAcceptorCount",
                        }
                        pubchem_field = prop_map.get(prop.lower(), prop)
                        val = p.get(pubchem_field)
                        if val is not None:
                            row["value"] = str(val)
                            row["source"] = f"PubChem (CID {cid})"
            except Exception as e:
                logger.debug(f"PubChem lookup failed for {smiles}: {e}")

            # If PubChem didn't have it, try ChEMBL
            if row["value"] == "N/A":
                try:
                    chembl_results = tu.tools.ChEMBL_search_similar_molecules(
                        query=smiles, similarity_threshold=100, max_results=1
                    )
                    # Extract from response (varies by format)
                    molecules = []
                    if isinstance(chembl_results, dict):
                        molecules = chembl_results.get("molecules", [])
                    elif isinstance(chembl_results, list):
                        molecules = chembl_results
                    if molecules:
                        mol = molecules[0] if isinstance(molecules[0], dict) else {}
                        mol_props = mol.get("molecule_properties", {})
                        if mol_props:
                            chembl_map = {"logP": "alogp", "mw": "full_mwt"}
                            val = mol_props.get(chembl_map.get(prop, prop))
                            if val is not None:
                                row["value"] = str(val)
                                row["source"] = f"ChEMBL ({mol.get('molecule_chembl_id', '')})"
                except Exception as e:
                    logger.debug(f"ChEMBL lookup failed for {smiles}: {e}")

            rows.append(row)
            time.sleep(0.3)  # Rate limit between molecules

        # Format as table
        header = f"Experimental validation for '{prop}':\n\n"
        header += f"{'SMILES':<40} | {'Value':<15} | {'Source'}\n"
        header += "-" * 80 + "\n"
        for r in rows:
            smiles_display = r["smiles"][:37] + "..." if len(r["smiles"]) > 40 else r["smiles"]
            header += f"{smiles_display:<40} | {r['value']:<15} | {r['source']}\n"

        found = sum(1 for r in rows if r["value"] != "N/A")
        header += f"\nFound experimental data for {found}/{len(rows)} molecules."

        return header
```

---

## Task 4: Update `src/pharma_agents/tools/__init__.py`

**File:** `src/pharma_agents/tools/__init__.py`
**Action:** MODIFY — add imports and `__all__` entries

Add these imports after the existing `from .skills import SkillLoaderTool` line:

```python
from .tooluniverse import (
    ADMETPredictTool,
    CompoundLookupTool,
    ExperimentalValidationTool,
    PubMedSearchTool,
)
```

Add these to the `__all__` list (after `"SkillLoaderTool"`):

```python
    # ToolUniverse
    "PubMedSearchTool",
    "CompoundLookupTool",
    "ADMETPredictTool",
    "ExperimentalValidationTool",
```

---

## Task 5: Wire tools to agents in `crew.py`

**File:** `src/pharma_agents/crew.py`
**Action:** MODIFY — 3 changes

### 5a. Add imports

Add to the imports from `.tools` (line 18-31):

```python
from .tools import (
    # ... existing imports ...
    PubMedSearchTool,
    CompoundLookupTool,
    ADMETPredictTool,
    ExperimentalValidationTool,
)
```

### 5b. Add `PubMedSearchTool` to archivist agent

In the `archivist_agent` method (~line 201), add `PubMedSearchTool()` to the tools list:

```python
tools=[
    SearchAndStoreTool(),
    PubMedSearchTool(),   # <-- ADD
    RemovePaperTool(),
],
```

### 5c. Add compound tools to hypothesis agent

In the `hypothesis_agent` method (~line 217), add to the tools list:

```python
tools=[
    ReadTrainPyTool(),
    LiteratureQueryTool(),
    SkillLoaderTool(),
    FetchMorePapersTool(),
    CompoundLookupTool(),    # <-- ADD
    ADMETPredictTool(),      # <-- ADD
],
```

### 5d. Add validation tool to evaluator agent

In the `evaluator_agent` method (~line 252), add to the tools list:

```python
tools=[
    ReadTrainPyTool(),
    RunTrainPyTool(),
    ExperimentalValidationTool(),  # <-- ADD
],
```

---

## Task 6: Update task prompts in `tasks.yaml`

**File:** `src/pharma_agents/tasks.yaml`
**Action:** MODIFY — 3 sections

### 6a. Archivist task — add PubMed step

After the existing SEARCH PHASE section (after the line "Do 3-5 searches. Each returns ~3 relevant papers."),
add a new sub-step:

```yaml
    3. PUBMED ENRICHMENT — After arxiv searches, run 1-2 PubMed searches
       using search_pubmed tool with queries like:
       - "{property} prediction machine learning"
       - "QSAR {property} deep learning"
       PubMed returns peer-reviewed articles (not just preprints).
       Note: PubMed results are for context only, they are NOT auto-stored
       in the literature database.
```

### 6b. Hypothesis task — add compound lookup guidance

After the existing "Based on the memory AND literature insights" paragraph (around line 68),
add these lines:

```yaml
    IF the dataset contains SMILES strings for specific molecules:
    - Use lookup_compound tool to check real experimental properties (logP, MW, TPSA)
      for 1-2 representative molecules from the dataset
    - Use predict_admet tool to get ADMET predictions if relevant to {property}
    - Use these real values to calibrate your hypothesis (e.g., feature ranges, thresholds)
```

### 6c. Evaluate task — add optional validation step

After the existing step 4 ("Recommend: KEEP or REVERT"), add:

```yaml
    5. (OPTIONAL) If the model improved, use validate_experimental to check
       predictions against real experimental values for 3-5 sample molecules.
       Input format: {{"smiles_list": ["CCO", "c1ccccc1"], "property": "logP"}}
       This adds confidence to the evaluation but is not required for the recommendation.
```

---

## Task 7: Enhance `SkillLoaderTool` in `skills.py`

**File:** `src/pharma_agents/tools/skills.py`
**Action:** MODIFY — 2 changes

### 7a. Add tooluniverse search path

In the `_run` method, after the existing `skill_paths` list (line 50-53), add a third search path:

```python
skill_paths = [
    SKILLS_DIR / "scientific" / f"{skill_name}.md",
    SKILLS_DIR / f"{skill_name}.md",
    SKILLS_DIR / skill_name / "SKILL.md",  # <-- ADD: for directory-based skills (tooluniverse-*)
]
```

### 7b. Update description

Update the `description` field to include tooluniverse skills:

```python
description: str = (
    "Loads a scientific skill to get domain knowledge and code examples. "
    "Available skills: rdkit, deepchem, datamol, molfeat, pytdc, "
    "chembl-database, pubchem-database, literature-review, "
    "tooluniverse-drug-research, tooluniverse-literature-deep-research, "
    "tooluniverse-chemical-compound-retrieval, tooluniverse-chemical-safety, "
    "tooluniverse-drug-repurposing, tooluniverse-target-research. "
    "Input: skill name (e.g., 'rdkit' or 'tooluniverse-drug-research'). "
    "Returns the skill content with best practices and workflow guides."
)
```

### 7c. Update available skills listing

In the fallback section at the bottom of `_run` (line 70-73), also list skills from
directory-based format:

```python
# List available skills
available = []
if (SKILLS_DIR / "scientific").exists():
    available = [f.stem for f in (SKILLS_DIR / "scientific").glob("*.md")]
# Also list directory-based skills (tooluniverse-*)
for skill_dir in SKILLS_DIR.iterdir():
    if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
        available.append(skill_dir.name)
return f"Skill '{skill_name}' not found. Available: {sorted(available)}"
```

---

## Task 8: Copy ToolUniverse skills to `.claude/skills/`

**Action:** Copy 15 skill directories from `/tmp/tu-skills/skills/` to `.claude/skills/`

Run these shell commands:

```bash
# Copy the 15 pharma-relevant skills
for skill in \
    tooluniverse \
    tooluniverse-drug-research \
    tooluniverse-literature-deep-research \
    tooluniverse-chemical-compound-retrieval \
    tooluniverse-drug-target-validation \
    tooluniverse-pharmacovigilance \
    tooluniverse-adverse-event-detection \
    tooluniverse-chemical-safety \
    tooluniverse-drug-repurposing \
    tooluniverse-network-pharmacology \
    tooluniverse-binder-discovery \
    tooluniverse-clinical-trial-design \
    tooluniverse-clinical-trial-matching \
    tooluniverse-drug-drug-interaction \
    tooluniverse-target-research; do
    cp -r "/tmp/tu-skills/skills/$skill" ".claude/skills/$skill"
done
```

---

## Task 9: Create `.claude/skills/scientific/tooluniverse.md`

**File:** `.claude/skills/scientific/tooluniverse.md`
**Action:** NEW (~100 lines)

This is a condensed overview mapping ToolUniverse skills to our BaseTool wrappers.

```markdown
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
```

---

## Task 10: Add tests to `tests/test_tools.py`

**File:** `tests/test_tools.py`
**Action:** MODIFY — add a new test class at the end (before `if __name__`)

**IMPORTANT:** Tests import from `pharma_agents.tools.custom_tools` — check the actual import path
used in existing tests. The existing tests use `from pharma_agents.tools.custom_tools import ...`.
If the module is actually at `pharma_agents.tools.tooluniverse`, use that path.
Check: the existing tests import `ArxivSearchTool` from `pharma_agents.tools.custom_tools`,
but the actual module is `pharma_agents.tools.arxiv`. There might be a `custom_tools.py` re-export.
Whatever import path the existing tests use, use the same pattern but import from the new module.

```python
class TestToolUniverseTools:
    """Test ToolUniverse-based tools against real APIs."""

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_pubmed_search_returns_results(self):
        """Search PubMed for a common pharma topic."""
        from pharma_agents.tools.tooluniverse import PubMedSearchTool

        PubMedSearchTool.reset_counters()
        tool = PubMedSearchTool()
        result = tool._run("molecular property prediction")

        assert "Found" in result
        assert "pmid:" in result or "s2:" in result

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_compound_lookup_with_ethanol(self):
        """Look up ethanol (simple, well-known molecule)."""
        from pharma_agents.tools.tooluniverse import CompoundLookupTool

        CompoundLookupTool.reset_counters()
        tool = CompoundLookupTool()
        result = tool._run("ethanol")

        assert "PubChem" in result
        assert "MW:" in result or "MolecularWeight" in result.lower()

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_compound_lookup_by_smiles(self):
        """Look up compound by SMILES string."""
        from pharma_agents.tools.tooluniverse import CompoundLookupTool

        CompoundLookupTool.reset_counters()
        tool = CompoundLookupTool()
        result = tool._run("CCO")  # ethanol SMILES

        assert "PubChem" in result

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_admet_predict_with_ethanol(self):
        """ADMET-AI prediction (may fail without tooluniverse[ml])."""
        from pharma_agents.tools.tooluniverse import ADMETPredictTool

        ADMETPredictTool.reset_counters()
        tool = ADMETPredictTool()
        result = tool._run('{"smiles": "CCO", "properties": ["physicochemical"]}')

        # Either returns predictions or a clear error about missing deps
        assert "ADMET-AI predictions" in result or "error" in result.lower()

    @pytest.mark.integration
    @pytest.mark.timeout(60)
    def test_experimental_validation_format(self):
        """Validate output format of experimental validation tool."""
        from pharma_agents.tools.tooluniverse import ExperimentalValidationTool

        ExperimentalValidationTool.reset_counters()
        tool = ExperimentalValidationTool()
        result = tool._run('{"smiles_list": ["CCO", "c1ccccc1"], "property": "logP"}')

        assert "Experimental validation" in result
        assert "SMILES" in result
        assert "Found experimental data for" in result

    @pytest.mark.integration
    @pytest.mark.timeout(15)
    def test_pubmed_search_rate_limit(self):
        """Verify max_calls_per_run limit."""
        from pharma_agents.tools.tooluniverse import PubMedSearchTool

        PubMedSearchTool.reset_counters()
        tool = PubMedSearchTool()
        tool.max_calls_per_run = 1

        tool._run("test query")
        result = tool._run("test query 2")
        assert "Max PubMed searches" in result
```

---

## Task 11: Update `docs/ARCHITECTURE.md`

**File:** `docs/ARCHITECTURE.md`
**Action:** MODIFY — add ToolUniverse section

### 11a. Add to Tech Stack table (after "Linting | ruff"):

```markdown
| ToolUniverse | tooluniverse SDK (Harvard Zitnik Lab, 1,996 tools) |
```

### 11b. Add to Tools Architecture table (after SkillLoaderTool row):

```markdown
| `PubMedSearchTool` | tooluniverse.py | Search PubMed + Semantic Scholar |
| `CompoundLookupTool` | tooluniverse.py | PubChem/ChEMBL compound properties |
| `ADMETPredictTool` | tooluniverse.py | ADMET-AI property predictions |
| `ExperimentalValidationTool` | tooluniverse.py | Validate predictions vs experimental data |
```

### 11c. Add new section after "Literature Pipeline" section:

```markdown
## ToolUniverse Integration

Wraps the ToolUniverse Python SDK (1,996 scientific tools) as CrewAI BaseTool subclasses.
Only ~4 tools are exposed to agents; 15 curated skill playbooks provide workflow guides.

```
┌──────────────────────────────────────────────────────────────────┐
│                   TOOLUNIVERSE PIPELINE                          │
│                                                                  │
│  Agent Tool Wrappers (CrewAI BaseTool)                          │
│  ├── PubMedSearchTool ──► PubMed + Semantic Scholar APIs        │
│  ├── CompoundLookupTool ──► PubChem + ChEMBL                   │
│  ├── ADMETPredictTool ──► ADMET-AI (9 endpoints)               │
│  └── ExperimentalValidationTool ──► PubChem + ChEMBL lookups   │
│                                                                  │
│  Shared ToolUniverse singleton (lazy-loaded, 1,996 tools)       │
│                                                                  │
│  Skill Playbooks (.claude/skills/tooluniverse-*/SKILL.md)       │
│  └── 15 curated pharma workflows loaded via SkillLoaderTool     │
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| BaseTool wrappers over raw SDK | Agents can't call raw Python — need CrewAI BaseTool interface |
| Curated 15 skills, not full 1,996 | Avoid broken/GPU-dependent tools; curated = reliable for demo |
| ADMET-AI as separate tool | Modularity — agent decides when to predict vs look up |
| Lazy singleton for ToolUniverse | ~2s init cost paid once, shared across all tool instances |
```

### 11d. Add to Environment Variables table:

```markdown
| `NCBI_API_KEY` | PubMed rate boost (3→10 req/s) | Optional |
```

---

## Task 12: Update `docs/decisions.md`

**File:** `docs/decisions.md`
**Action:** MODIFY — append new ADR

```markdown

## 2026-03-16 — ToolUniverse Integration via Local Skill Curation

**Status:** accepted
**Context:** The demo only searches arxiv for literature and has no way to validate predictions
against real experimental data. ToolUniverse (Harvard Zitnik Lab) provides 1,996 scientific tools
via a Python SDK, including PubMed, Semantic Scholar, PubChem, ChEMBL, and ADMET-AI.
**Decision:** Wrap ~4 ToolUniverse SDK calls as CrewAI BaseTool subclasses. Copy 15 pharma-relevant
skill playbooks to `.claude/skills/` for agent workflow guidance. Use a lazy module-level singleton
for the ToolUniverse instance. Do NOT use the full SDK discovery (1,996 tools) at agent runtime.
**Alternatives considered:**
- Full SDK discovery at runtime — rejected because many tools require GPU dependencies or specific
  API keys, making the demo fragile. Curating 15 skills ensures reliability.
- Direct API calls without ToolUniverse SDK — rejected because the SDK handles auth, rate limiting,
  and response parsing. Wrapping the SDK is less code and more maintainable.
- MCP server for agent access — rejected for autonomous agents (MCP is for interactive Claude Code
  use). BaseTool wrappers are the correct CrewAI integration pattern.
**Consequences:**
- New dependency: `tooluniverse>=1.0.0` (~2s load time on first use).
- ADMET-AI predictions require `tooluniverse[ml]` extra (heavy deps); graceful fallback if missing.
- 15 skill directories added to `.claude/skills/` (~500KB total).
- Demo gains: multi-source literature, experimental data lookups, ADMET predictions, validation.
```

---

## Task 13: Update MCP settings (`.claude/settings.local.json`)

**File:** `.claude/settings.local.json`
**Action:** MODIFY — add tooluniverse MCP server permission

Add to the `allow` array:

```json
"Bash(uvx tooluniverse:*)"
```

This allows Claude Code to run ToolUniverse CLI commands interactively (e.g., `uvx tooluniverse tu find "solubility"`).

---

## Execution Order

Tasks are independent enough to run in small batches:

1. **Task 1** (pyproject.toml) → run `uv sync` to verify
2. **Task 2** (.env.example) — trivial
3. **Task 3** (tooluniverse.py) — the main new file
4. **Task 4** (__init__.py) — depends on Task 3
5. **Task 5** (crew.py) — depends on Task 4
6. **Task 6** (tasks.yaml) — independent
7. **Task 7** (skills.py) — independent
8. **Task 8** (copy skills) — shell command, independent
9. **Task 9** (scientific/tooluniverse.md) — independent
10. **Task 10** (tests) — depends on Task 3
11. **Task 11** (ARCHITECTURE.md) — independent
12. **Task 12** (decisions.md) — independent
13. **Task 13** (settings) — independent

## Verification Checklist

After all tasks:

1. `uv sync` — tooluniverse installs cleanly
2. `uv run ruff check src/pharma_agents/tools/tooluniverse.py` — no lint errors
3. `uv run pytest tests/test_tools.py -v -k "not integration"` — existing 14 tests still pass
4. `uv run pytest tests/test_tools.py -v -m integration` — new ToolUniverse tests pass (needs network)
5. `uv run python -c "from pharma_agents.tools import PubMedSearchTool; print('OK')"` — imports work
6. Skill test: verify `.claude/skills/tooluniverse-drug-research/SKILL.md` exists and is readable
