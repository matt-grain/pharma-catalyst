"""
Canned LLM responses for the mock server.

Each response is tagged with a 'context' hint so the server can suggest
the most relevant canned response based on the incoming request's content.

To add new canned responses: just append to CANNED_RESPONSES list.
"""

from __future__ import annotations

import json
from typing import TypedDict


class CannedResponse(TypedDict):
    id: str
    label: str
    context_hint: str  # regex or keyword matched against incoming messages
    response: str


CANNED_RESPONSES: list[CannedResponse] = [
    # ── Hypothesis Agent ──────────────────────────────────────────────
    {
        "id": "hypothesis_1",
        "label": "Hypothesis: Add RDKit descriptors",
        "context_hint": "hypothesis|proposal|Research Scientist|improvement",
        "response": json.dumps(
            {
                "proposal": "Add RDKit molecular descriptors (LogP, TPSA, MolWt, NumHDonors, NumHAcceptors, NumRotatableBonds) as features alongside Morgan fingerprints",
                "reasoning": "Morgan fingerprints capture substructural patterns but miss global molecular properties. Adding physicochemical descriptors provides complementary information about drug-likeness and membrane permeability that are known to correlate with toxicity endpoints.",
                "change_description": "In train.py, after computing Morgan fingerprints, compute 6 RDKit descriptors per molecule and concatenate them as additional feature columns.",
                "literature_insight": "Multiple QSAR studies show that combining fingerprints with physicochemical descriptors improves prediction accuracy for ADMET endpoints (Mayr et al. 2016, Wu et al. 2018).",
            },
            indent=2,
        ),
    },
    {
        "id": "hypothesis_2",
        "label": "Hypothesis: Switch to Random Forest",
        "context_hint": "hypothesis|proposal|Research Scientist",
        "response": json.dumps(
            {
                "proposal": "Replace the current model with a Random Forest classifier with 500 trees and class_weight='balanced'",
                "reasoning": "The current model may underperform on imbalanced toxicity datasets. Random Forest with balanced class weights handles class imbalance natively and provides feature importance for interpretability.",
                "change_description": "In train.py, replace the model initialization with RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1).",
                "literature_insight": "Random Forests remain competitive baselines for molecular property prediction, especially with tabular features (Sheridan 2013).",
            },
            indent=2,
        ),
    },
    # ── Review Panel: Statistician ────────────────────────────────────
    {
        "id": "statistician_approve",
        "label": "Statistician: Approve with notes",
        "context_hint": "Statistician|statistical|sample size",
        "response": (
            "The proposal is statistically reasonable. Adding 6 physicochemical descriptors "
            "to ~1,400 ClinTox samples gives a features-to-samples ratio that remains safe "
            "for tree-based models (no overfitting concern). The 5-fold stratified CV already "
            "in place will give a reliable estimate of improvement. However, I'd recommend "
            "checking feature correlation — if TPSA and LogP are highly correlated with "
            "existing fingerprint bits, the marginal gain may be minimal."
        ),
    },
    {
        "id": "statistician_concern",
        "label": "Statistician: Raise overfitting concern",
        "context_hint": "Statistician|statistical",
        "response": (
            "I have concerns about overfitting. The current dataset has ~1,400 samples, and "
            "adding complex features without regularization increases the risk of memorizing "
            "the training set. The proposed change doesn't mention any regularization strategy. "
            "I'd recommend adding L2 regularization or using a model with built-in "
            "regularization (e.g., Random Forest with max_depth limits)."
        ),
    },
    # ── Review Panel: Medicinal Chemist ───────────────────────────────
    {
        "id": "chemist_approve",
        "label": "Chemist: Strong approval",
        "context_hint": "Medicinal_Chemist|chemical|molecular|SAR",
        "response": (
            "From a medicinal chemistry perspective, this proposal is well-grounded. LogP and "
            "TPSA are fundamental descriptors in toxicity prediction — they directly relate to "
            "membrane permeability and oral bioavailability, both of which correlate with "
            "toxicity outcomes. The Lipinski descriptors (HBD, HBA, MW, RotBonds) capture "
            "drug-likeness properties that the fingerprints miss. This is exactly the kind of "
            "domain-informed feature engineering I'd recommend."
        ),
    },
    # ── Review Panel: Devil's Advocate ────────────────────────────────
    {
        "id": "devil_mild",
        "label": "Devil's Advocate: Mild concerns",
        "context_hint": "Devil|flaw|weakness|pitfall",
        "response": (
            "The strongest argument against this proposal: it's incremental, not innovative. "
            "Adding 6 standard descriptors is the most obvious thing to try, and if the "
            "baseline fingerprints already capture this information implicitly, the improvement "
            "will be negligible. Also, the implementation risk is low but nonzero — incorrect "
            "RDKit descriptor computation (e.g., failing on invalid SMILES) could silently "
            "produce NaN features. Ensure proper error handling for invalid molecules."
        ),
    },
    {
        "id": "devil_harsh",
        "label": "Devil's Advocate: Strong rejection push",
        "context_hint": "Devil|flaw|weakness",
        "response": (
            "This proposal has fundamental issues. First, the 6 proposed descriptors are "
            "likely redundant with Morgan fingerprint bits — LogP correlates with hydrophobic "
            "substructures already encoded. Second, concatenating heterogeneous features "
            "(binary fingerprints + continuous descriptors) without normalization will cause "
            "tree splits to favor the continuous features, effectively ignoring the fingerprints. "
            "Third, no feature selection is proposed — adding features without removing noisy "
            "ones is a recipe for overfitting on 1,400 samples."
        ),
    },
    # ── Review Panel: Team Memory Analyst ─────────────────────────────
    {
        "id": "memory_novel",
        "label": "Memory Analyst: Novel approach",
        "context_hint": "Team_Memory|experiment history|previous|tried before",
        "response": (
            "Reviewing the experiment history: this approach has NOT been tried before. "
            "Previous iterations focused on model architecture changes (iterations 1-2) and "
            "hyperparameter tuning (iteration 3). Feature engineering with RDKit descriptors "
            "is a genuinely new direction. This is good — it expands the search space rather "
            "than circling the same design decisions."
        ),
    },
    {
        "id": "memory_repeat",
        "label": "Memory Analyst: Duplicate detected",
        "context_hint": "Team_Memory|experiment history",
        "response": (
            "WARNING: This is a near-duplicate of iteration 2, which also attempted to add "
            "physicochemical descriptors. That attempt achieved ROC-AUC 0.872 vs baseline "
            "0.865 — a marginal improvement within noise. The current proposal adds the same "
            "core descriptors (LogP, TPSA, MW). Unless the implementation approach is "
            "fundamentally different, we should expect similar marginal gains."
        ),
    },
    # ── Review Panel: Pharma Ethics ───────────────────────────────────
    {
        "id": "ethics_ok",
        "label": "Ethics: No concerns",
        "context_hint": "Pharma_Ethics|regulatory|FDA|interpretab",
        "response": (
            "No significant ethics or regulatory concerns with this proposal. The added "
            "descriptors are well-established physicochemical properties with clear physical "
            "meaning, which supports model interpretability. The approach uses standard "
            "cross-validation, maintaining proper train/test separation for GxP compliance. "
            "The deterministic descriptor computation ensures reproducibility."
        ),
    },
    # ── Review Panel: Moderator Verdicts ──────────────────────────────
    {
        "id": "moderator_approve",
        "label": "Moderator: APPROVE",
        "context_hint": "Moderator|verdict|summarize|decision",
        "response": json.dumps(
            {
                "decision": "approved",
                "feedback": "The panel finds this proposal well-grounded. The Statistician confirms adequate sample size, the Chemist validates the domain relevance of proposed descriptors, and the Devil's Advocate raised only minor implementation concerns. Proceed with careful handling of invalid SMILES.",
                "confidence": 0.82,
                "concerns": [
                    "Ensure NaN handling for invalid SMILES in descriptor computation",
                    "Monitor feature correlation to confirm descriptors add non-redundant information",
                ],
            },
            indent=2,
        ),
    },
    {
        "id": "moderator_revise",
        "label": "Moderator: REVISE",
        "context_hint": "Moderator|verdict|summarize|decision",
        "response": json.dumps(
            {
                "decision": "revised",
                "revised_proposal": "Add RDKit descriptors BUT normalize all features to [0,1] range before concatenation, and add a feature correlation check to drop descriptors with >0.95 correlation to existing fingerprint features.",
                "revised_reasoning": "The Devil's Advocate correctly identified that mixing binary and continuous features without normalization biases tree-based models. Adding normalization addresses this while preserving the chemically sound feature set.",
                "feedback": "The core idea is chemically sound but needs normalization and correlation checking to avoid the pitfalls identified by the Devil's Advocate.",
                "confidence": 0.75,
                "concerns": [
                    "Raw concatenation of binary and continuous features causes scale bias",
                    "Possible redundancy with existing fingerprint bits",
                ],
            },
            indent=2,
        ),
    },
    {
        "id": "moderator_reject",
        "label": "Moderator: REJECT",
        "context_hint": "Moderator|verdict|summarize|decision",
        "response": json.dumps(
            {
                "decision": "rejected",
                "feedback": "The panel identified fundamental issues: this is a near-duplicate of a previously tried approach that yielded negligible improvement, and the implementation lacks normalization and feature selection. The team should explore a genuinely different direction.",
                "confidence": 0.85,
                "concerns": [
                    "Near-duplicate of iteration 2 with marginal results",
                    "No normalization strategy for heterogeneous features",
                    "No feature selection to manage curse of dimensionality",
                ],
            },
            indent=2,
        ),
    },
    # ── Implementation Agent ──────────────────────────────────────────
    {
        "id": "impl_read_train",
        "label": "Implementation: Read train.py first",
        "context_hint": "implement|train.py|read|current code",
        "response": "I'll start by reading the current train.py to understand the existing feature pipeline before making changes.\n\nAction: read_train_py\nArgument: read",
    },
    {
        "id": "impl_write_code",
        "label": "Implementation: Write RDKit descriptors code",
        "context_hint": "implement|write|modify|change",
        "response": (
            "I'll now modify train.py to add RDKit descriptors.\n\n"
            "Action: write_code\n"
            "Argument: \n"
            "```python\n"
            "# After Morgan fingerprint computation, add:\n"
            "from rdkit.Chem import Descriptors\n"
            "desc_names = ['MolLogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds']\n"
            "desc_funcs = [getattr(Descriptors, d) for d in desc_names]\n"
            "descriptors = np.array([[f(mol) if mol else 0.0 for f in desc_funcs] for mol in mols])\n"
            "X = np.hstack([X_fp, descriptors])\n"
            "```"
        ),
    },
    {
        "id": "impl_run_train",
        "label": "Implementation: Run training",
        "context_hint": "implement|run|train|execute",
        "response": "Training complete. Let me run the model now.\n\nAction: run_train_py\nArgument: run",
    },
    # ── Evaluator Agent ───────────────────────────────────────────────
    {
        "id": "eval_improved",
        "label": "Evaluator: Score improved",
        "context_hint": "evaluat|score|metric|ROC|AUC|improvement",
        "response": (
            "## Evaluation Results\n\n"
            "**Previous ROC-AUC:** 0.8650\n"
            "**New ROC-AUC:** 0.8923\n"
            "**Improvement:** +0.0273 (+3.16%)\n\n"
            "The addition of RDKit descriptors produced a meaningful improvement. "
            "The 5-fold CV standard deviation is 0.015, so the improvement exceeds "
            "one standard deviation — likely a genuine signal rather than noise.\n\n"
            "RESULT: improved\n"
            "SCORE: 0.8923"
        ),
    },
    {
        "id": "eval_no_change",
        "label": "Evaluator: No improvement",
        "context_hint": "evaluat|score|metric",
        "response": (
            "## Evaluation Results\n\n"
            "**Previous ROC-AUC:** 0.8650\n"
            "**New ROC-AUC:** 0.8672\n"
            "**Improvement:** +0.0022 (+0.25%)\n\n"
            "The change produced a negligible improvement within noise margins. "
            "The 5-fold CV standard deviation is 0.018, so this delta is not "
            "statistically significant.\n\n"
            "RESULT: no_improvement\n"
            "SCORE: 0.8672"
        ),
    },
]


def match_canned_responses(messages: list[dict]) -> list[CannedResponse]:
    """Return canned responses ranked by relevance to the incoming messages."""
    import re

    # Build a text blob from all messages for matching
    text = " ".join(m.get("content", "") for m in messages).lower()
    # Also check system messages for agent names
    text += " " + " ".join(m.get("name", "") for m in messages).lower()

    scored: list[tuple[int, CannedResponse]] = []
    for canned in CANNED_RESPONSES:
        hints = canned["context_hint"].split("|")
        score = sum(1 for hint in hints if re.search(hint.lower(), text))
        if score > 0:
            scored.append((score, canned))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]
