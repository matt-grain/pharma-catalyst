"""PubMed, PubChem, and ToolUniverse tools via direct REST APIs.

Wraps NCBI E-utilities, PubChem PUG REST, and ToolUniverse catalog API
as CrewAI BaseTool subclasses. No heavy SDK dependency — just urllib calls.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import ClassVar

from crewai.tools import BaseTool
from loguru import logger


def _fetch_json(url: str, timeout: float = 15.0) -> dict | list | None:
    """Fetch JSON from a URL with error handling."""
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "pharma-agents/0.1")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.debug(f"HTTP fetch failed for {url[:80]}: {e}")
        return None


class PubMedSearchTool(BaseTool):
    """Search PubMed for peer-reviewed biomedical papers."""

    name: str = "search_pubmed"
    description: str = (
        "Searches PubMed for peer-reviewed papers on a topic. "
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

    def _search_pubmed(self, query: str, max_results: int = 3) -> list[dict]:
        """Search PubMed via NCBI E-utilities (esearch + esummary)."""
        import os

        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        api_key = os.getenv("NCBI_API_KEY", "")
        key_param = f"&api_key={api_key}" if api_key else ""

        # Step 1: esearch — get PMIDs
        search_url = (
            f"{base}/esearch.fcgi?db=pubmed&retmode=json&retmax={max_results}"
            f"&sort=relevance&term={urllib.parse.quote(query)}{key_param}"
        )
        search_data = _fetch_json(search_url)
        if not search_data or "esearchresult" not in search_data:
            return []

        pmids = search_data["esearchresult"].get("idlist", [])
        if not pmids:
            return []

        # Step 2: esummary — get metadata for PMIDs
        ids_str = ",".join(pmids)
        summary_url = (
            f"{base}/esummary.fcgi?db=pubmed&retmode=json&id={ids_str}{key_param}"
        )
        summary_data = _fetch_json(summary_url)
        if not summary_data or "result" not in summary_data:
            return [{"pmid": p, "title": f"PMID {p}", "abstract": ""} for p in pmids]

        results = []
        for pmid in pmids:
            article = summary_data["result"].get(pmid, {})
            results.append(
                {
                    "pmid": pmid,
                    "title": article.get("title", ""),
                    "authors": [
                        a.get("name", "") for a in article.get("authors", [])[:3]
                    ],
                    "journal": article.get("fulljournalname", ""),
                    "pub_date": article.get("pubdate", ""),
                    "doi": article.get("elocationid", ""),
                }
            )

        # Step 3: efetch — get abstracts (optional enrichment)
        try:
            fetch_url = (
                f"{base}/efetch.fcgi?db=pubmed&retmode=xml&id={ids_str}{key_param}"
            )
            req = urllib.request.Request(fetch_url)
            req.add_header("User-Agent", "pharma-agents/0.1")
            with urllib.request.urlopen(req, timeout=15) as resp:
                xml_content = resp.read().decode("utf-8")

            import xml.etree.ElementTree as ET

            root = ET.fromstring(xml_content)
            for article_el in root.findall(".//PubmedArticle"):
                pmid_el = article_el.find(".//PMID")
                abstract_el = article_el.find(".//AbstractText")
                if pmid_el is not None and abstract_el is not None:
                    pmid_text = pmid_el.text or ""
                    for r in results:
                        if r["pmid"] == pmid_text:
                            r["abstract"] = abstract_el.text or ""
                            break
        except Exception as e:
            logger.debug(f"PubMed abstract fetch failed: {e}")

        return results

    def _run(self, query: str) -> str:
        if PubMedSearchTool._calls_done >= self.max_calls_per_run:
            return f"Max PubMed searches ({self.max_calls_per_run}) reached."

        # Rate limiting
        elapsed = time.time() - PubMedSearchTool._last_call
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        PubMedSearchTool._last_call = time.time()
        PubMedSearchTool._calls_done += 1

        results = self._search_pubmed(query)

        if not results:
            return f"No papers found for query: {query}"

        formatted = []
        for r in results:
            abstract = r.get("abstract", "")
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            authors = ", ".join(r.get("authors", []))
            formatted.append(
                f"- **pmid:{r['pmid']}** ({r.get('pub_date', '')}): {r['title']}\n"
                f"  Authors: {authors}\n"
                f"  Journal: {r.get('journal', '')}\n"
                f"  Abstract: {abstract}\n"
            )

        return f"Found {len(results)} PubMed papers for '{query}':\n\n" + "\n".join(
            formatted
        )


class CompoundLookupTool(BaseTool):
    """Look up compound properties from PubChem."""

    name: str = "lookup_compound"
    description: str = (
        "Looks up real experimental properties for a molecule from PubChem. "
        "Input: SMILES string (e.g., 'CCO' for ethanol) or compound name (e.g., 'aspirin'). "
        "Returns molecular weight, logP, TPSA, hydrogen bond counts, and SMILES. "
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

    def _resolve_cid(self, compound: str) -> int | None:
        """Resolve a compound name or SMILES to a PubChem CID."""
        base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

        # Try as SMILES first (if it looks like one)
        is_smiles = any(c in compound for c in "()=#[]") or (
            len(compound) < 100
            and " " not in compound
            and len(compound) > 1
            and not compound[0].isdigit()
        )

        if is_smiles:
            url = f"{base}/compound/smiles/{urllib.parse.quote(compound, safe='')}/cids/JSON"
            data = _fetch_json(url)
            if isinstance(data, dict) and "IdentifierList" in data:
                cids = data["IdentifierList"].get("CID", [])
                if cids and cids[0] != 0:
                    return cids[0]

        # Try as name
        url = f"{base}/compound/name/{urllib.parse.quote(compound)}/cids/JSON"
        data = _fetch_json(url)
        if isinstance(data, dict) and "IdentifierList" in data:
            cids = data["IdentifierList"].get("CID", [])
            if cids and cids[0] != 0:
                return cids[0]

        return None

    def _get_properties(self, cid: int) -> dict | None:
        """Get compound properties from PubChem PUG REST."""
        props = "MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,CanonicalSMILES,MolecularFormula"
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{props}/JSON"
        data = _fetch_json(url)
        if isinstance(data, dict) and "PropertyTable" in data:
            prop_list = data["PropertyTable"].get("Properties", [])
            if prop_list:
                return prop_list[0]
        return None

    def _run(self, compound: str) -> str:
        if CompoundLookupTool._calls_done >= self.max_calls_per_run:
            return f"Max compound lookups ({self.max_calls_per_run}) reached."

        elapsed = time.time() - CompoundLookupTool._last_call
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        CompoundLookupTool._last_call = time.time()
        CompoundLookupTool._calls_done += 1

        compound = compound.strip()
        cid = self._resolve_cid(compound)
        if cid is None:
            return f"Could not find compound '{compound}' in PubChem."

        props = self._get_properties(cid)
        if not props:
            return f"Found CID {cid} but could not retrieve properties."

        return (
            f"Compound lookup for '{compound}':\n\n"
            f"**PubChem (CID {cid}):**\n"
            f"  SMILES: {props.get('CanonicalSMILES', 'N/A')}\n"
            f"  Formula: {props.get('MolecularFormula', 'N/A')}\n"
            f"  MW: {props.get('MolecularWeight', 'N/A')}\n"
            f"  LogP (XLogP3): {props.get('XLogP', 'N/A')}\n"
            f"  TPSA: {props.get('TPSA', 'N/A')}\n"
            f"  HBD: {props.get('HBondDonorCount', 'N/A')}, "
            f"HBA: {props.get('HBondAcceptorCount', 'N/A')}\n"
            f"  Rotatable bonds: {props.get('RotatableBondCount', 'N/A')}"
        )


class ExperimentalValidationTool(BaseTool):
    """Validate predictions against experimental data from PubChem."""

    name: str = "validate_experimental"
    description: str = (
        "Looks up experimental values for molecules to validate model predictions. "
        "Input: JSON with 'smiles_list' (up to 10 SMILES) and 'property' "
        "(e.g., 'logP', 'mw', 'tpsa', 'hbd', 'hba'). "
        "Returns a comparison table: SMILES | Experimental Value | Source. "
        "Use this to ground ML predictions in real-world data."
    )
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

    max_calls_per_run: int = 3
    _calls_done: ClassVar[int] = 0

    @classmethod
    def reset_counters(cls) -> None:
        cls._calls_done = 0

    # Map user-friendly property names to PubChem property fields
    _PROP_MAP: ClassVar[dict[str, str]] = {
        "logp": "XLogP",
        "logP": "XLogP",
        "mw": "MolecularWeight",
        "molecular_weight": "MolecularWeight",
        "tpsa": "TPSA",
        "hbd": "HBondDonorCount",
        "hba": "HBondAcceptorCount",
        "rotatable_bonds": "RotatableBondCount",
    }

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

        pubchem_field = self._PROP_MAP.get(prop, prop)
        lookup = CompoundLookupTool()
        rows = []

        for smiles in smiles_list:
            row = {"smiles": smiles, "value": "N/A", "source": "none"}

            try:
                cid = lookup._resolve_cid(smiles)
                if cid:
                    props = lookup._get_properties(cid)
                    if props:
                        val = props.get(pubchem_field)
                        if val is not None:
                            row["value"] = str(val)
                            row["source"] = f"PubChem (CID {cid})"
            except Exception as e:
                logger.debug(f"Validation lookup failed for {smiles}: {e}")

            rows.append(row)
            time.sleep(0.3)  # Rate limit between molecules

        # Format as table
        header = f"Experimental validation for '{prop}':\n\n"
        header += f"{'SMILES':<40} | {'Value':<15} | {'Source'}\n"
        header += "-" * 80 + "\n"
        for r in rows:
            smiles_display = (
                r["smiles"][:37] + "..." if len(r["smiles"]) > 40 else r["smiles"]
            )
            header += f"{smiles_display:<40} | {r['value']:<15} | {r['source']}\n"

        found = sum(1 for r in rows if r["value"] != "N/A")
        header += f"\nFound experimental data for {found}/{len(rows)} molecules."

        return header


# Cached ToolUniverse catalog (fetched once per process)
_tu_catalog: list[dict] | None = None

# Full catalog with toolType, isValidated, tags, category, source
_TU_CATALOG_URL = "https://aiscientist.tools/tools_restored.json"


def _get_tu_catalog() -> list[dict]:
    """Fetch and cache the ToolUniverse tool catalog (1,900+ tools)."""
    global _tu_catalog  # noqa: PLW0603
    if _tu_catalog is None:
        data = _fetch_json(_TU_CATALOG_URL, timeout=30.0)
        _tu_catalog = data if isinstance(data, list) else []
        logger.info(f"ToolUniverse catalog: {len(_tu_catalog)} tools loaded")
    return _tu_catalog


class ToolUniverseSearchTool(BaseTool):
    """Search the ToolUniverse catalog for validated scientific tools and ML models."""

    name: str = "search_tooluniverse"
    description: str = (
        "Searches the ToolUniverse catalog (1900+ scientific tools from Harvard Zitnik Lab) "
        "for validated ML models and tools relevant to a topic. "
        "Input: keyword query (e.g., 'solubility prediction', 'BBB penetrance', 'toxicity'). "
        "Returns tool names, types (ML Model/API/AI Agent), descriptions, and validation status. "
        "Use this to discover what validated approaches exist for a prediction task."
    )
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

    max_calls_per_run: int = 3
    _calls_done: ClassVar[int] = 0

    @classmethod
    def reset_counters(cls) -> None:
        cls._calls_done = 0

    def _run(self, query: str) -> str:
        if ToolUniverseSearchTool._calls_done >= self.max_calls_per_run:
            return f"Max ToolUniverse searches ({self.max_calls_per_run}) reached."
        ToolUniverseSearchTool._calls_done += 1

        catalog = _get_tu_catalog()
        if not catalog:
            return "Error: Could not fetch ToolUniverse catalog."

        # Client-side keyword search across name + description + tags + category
        keywords = [k.lower() for k in query.strip().split() if k.strip()]
        if not keywords:
            return "Error: Empty search query."

        scored: list[tuple[int, dict]] = []
        for tool in catalog:
            name = tool.get("name", "").lower()
            desc = tool.get("description", "").lower()
            tags = " ".join(tool.get("tags", [])).lower()
            category = tool.get("category", "").lower()
            source = tool.get("source", "").lower()
            searchable = f"{name} {desc} {tags} {category} {source}"

            hits = sum(1 for kw in keywords if kw in searchable)
            if hits > 0:
                # Boost validated tools and ML models
                bonus = 0
                if tool.get("isValidated"):
                    bonus += 1
                if tool.get("toolType") == "ML Model":
                    bonus += 1
                scored.append((hits + bonus, tool))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:8]

        if not top:
            return f"No tools found matching '{query}' in ToolUniverse catalog."

        lines = [
            f"Found {len(scored)} ToolUniverse tools matching '{query}' (showing top {len(top)}):\n"
        ]
        for _score, t in top:
            name = t.get("name", "")
            tool_type = t.get("toolType", "")
            validated = " Validated" if t.get("isValidated") else ""
            desc = t.get("description", "")
            if len(desc) > 200:
                desc = desc[:200] + "..."
            tags = ", ".join(t.get("tags", []))
            source = t.get("source", "")
            params = t.get("parameters", {})
            param_names = list(params.keys()) if isinstance(params, dict) else []
            param_str = f" | Params: {', '.join(param_names)}" if param_names else ""
            lines.append(
                f"- **{name}** [{tool_type}{validated}] ({tags})\n"
                f"  Source: {source}\n"
                f"  {desc}{param_str}\n"
            )

        lines.append(
            "These are validated tools from the ToolUniverse ecosystem (Harvard Zitnik Lab). "
            "Use their approaches and methods as inspiration for your hypothesis."
        )
        return "\n".join(lines)
