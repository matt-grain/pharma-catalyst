#!/usr/bin/env python3
"""
Benchmark embedding models for scientific paper retrieval.

Compares:
- fastembed (ONNX, CPU-optimized)
- sentence-transformers (PyTorch)
- Domain-specific models (SPECTER2, SciBERT)

Usage:
    uv run python scripts/benchmark_embeddings.py
"""

import time
from dataclasses import dataclass

# Scientific paper examples (molecular ML domain)
TEST_QUERIES = [
    "ADMET property prediction using graph neural networks",
    "Blood-brain barrier permeability machine learning",
    "Morgan fingerprints molecular featurization",
    "Solubility prediction deep learning",
]

TEST_DOCUMENTS = [
    "We present a novel approach to ADMET prediction using message-passing neural networks on molecular graphs.",
    "Blood-brain barrier penetration can be predicted using random forests with physicochemical descriptors.",
    "Morgan fingerprints encode circular substructures and are widely used in cheminformatics applications.",
    "Deep learning models outperform traditional methods for aqueous solubility prediction on ESOL dataset.",
    "Graph attention networks show improved performance on molecular property prediction benchmarks.",
    "The BBBP dataset contains 2039 compounds with binary blood-brain barrier permeability labels.",
    "Gradient boosting with RDKit descriptors achieves state-of-the-art results on MoleculeNet.",
    "Transformer architectures can learn molecular representations directly from SMILES strings.",
    "Feature engineering with physicochemical properties remains competitive with deep learning.",
    "Multi-task learning improves generalization across ADMET endpoints.",
]


@dataclass
class BenchmarkResult:
    model_name: str
    load_time_ms: float
    embed_query_ms: float
    embed_batch_ms: float
    total_ms: float
    retrieval_quality: float  # Mean reciprocal rank


def benchmark_fastembed() -> BenchmarkResult:
    """Benchmark fastembed with BGE-small."""
    print("\n[fastembed] BAAI/bge-small-en-v1.5")

    # Load
    start = time.perf_counter()
    from fastembed import TextEmbedding

    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    # Warm up
    list(model.embed(["warmup"]))
    load_time = (time.perf_counter() - start) * 1000
    print(f"  Load time: {load_time:.0f}ms")

    # Embed single query
    start = time.perf_counter()
    for _ in range(10):
        list(model.embed([TEST_QUERIES[0]]))
    embed_query = (time.perf_counter() - start) * 100  # avg per query
    print(f"  Embed query (avg): {embed_query:.1f}ms")

    # Embed batch
    start = time.perf_counter()
    doc_embeddings = list(model.embed(TEST_DOCUMENTS))
    embed_batch = (time.perf_counter() - start) * 1000
    print(f"  Embed batch ({len(TEST_DOCUMENTS)} docs): {embed_batch:.0f}ms")

    # Retrieval quality (simple cosine similarity check)
    import numpy as np

    query_emb = list(model.embed([TEST_QUERIES[0]]))[0]

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sims = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Best match should be doc 0 (ADMET + GNN)
    rank = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True).index(0) + 1
    mrr = 1.0 / rank
    print(f"  Retrieval quality (MRR): {mrr:.2f}")

    return BenchmarkResult(
        model_name="fastembed/bge-small",
        load_time_ms=load_time,
        embed_query_ms=embed_query,
        embed_batch_ms=embed_batch,
        total_ms=load_time + embed_batch,
        retrieval_quality=mrr,
    )


def benchmark_sentence_transformers(
    model_name: str = "all-MiniLM-L6-v2",
) -> BenchmarkResult:
    """Benchmark sentence-transformers."""
    print(f"\n[sentence-transformers] {model_name}")

    try:
        # Load
        start = time.perf_counter()
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        # Warm up
        model.encode(["warmup"])
        load_time = (time.perf_counter() - start) * 1000
        print(f"  Load time: {load_time:.0f}ms")

        # Embed single query
        start = time.perf_counter()
        for _ in range(10):
            model.encode([TEST_QUERIES[0]])
        embed_query = (time.perf_counter() - start) * 100
        print(f"  Embed query (avg): {embed_query:.1f}ms")

        # Embed batch
        start = time.perf_counter()
        doc_embeddings = model.encode(TEST_DOCUMENTS)
        embed_batch = (time.perf_counter() - start) * 1000
        print(f"  Embed batch ({len(TEST_DOCUMENTS)} docs): {embed_batch:.0f}ms")

        # Retrieval quality
        import numpy as np

        query_emb = model.encode([TEST_QUERIES[0]])[0]

        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sims = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
        rank = (
            sorted(range(len(sims)), key=lambda i: sims[i], reverse=True).index(0) + 1
        )
        mrr = 1.0 / rank
        print(f"  Retrieval quality (MRR): {mrr:.2f}")

        return BenchmarkResult(
            model_name=f"st/{model_name}",
            load_time_ms=load_time,
            embed_query_ms=embed_query,
            embed_batch_ms=embed_batch,
            total_ms=load_time + embed_batch,
            retrieval_quality=mrr,
        )
    except ImportError:
        print("  SKIPPED (sentence-transformers not installed)")
        return BenchmarkResult(
            model_name=f"st/{model_name}",
            load_time_ms=0,
            embed_query_ms=0,
            embed_batch_ms=0,
            total_ms=0,
            retrieval_quality=0,
        )


def benchmark_specter2() -> BenchmarkResult:
    """Benchmark SPECTER2 (scientific paper embeddings)."""
    return benchmark_sentence_transformers("allenai/specter2_base")


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print comparison summary."""
    print("\n" + "=" * 70)
    print("EMBEDDING MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(
        f"{'Model':<30} {'Load(ms)':<10} {'Query(ms)':<10} {'Batch(ms)':<10} {'MRR':<6}"
    )
    print("-" * 70)

    for r in results:
        if r.total_ms > 0:  # Skip skipped models
            print(
                f"{r.model_name:<30} {r.load_time_ms:<10.0f} {r.embed_query_ms:<10.1f} {r.embed_batch_ms:<10.0f} {r.retrieval_quality:<6.2f}"
            )

    print("=" * 70)

    # Recommendation
    valid = [r for r in results if r.total_ms > 0]
    if valid:
        fastest = min(valid, key=lambda r: r.embed_query_ms)
        best_quality = max(valid, key=lambda r: r.retrieval_quality)

        print(f"\nFastest: {fastest.model_name} ({fastest.embed_query_ms:.1f}ms/query)")
        print(
            f"Best retrieval: {best_quality.model_name} (MRR={best_quality.retrieval_quality:.2f})"
        )

        # If same model, clear winner
        if fastest.model_name == best_quality.model_name:
            print(
                f"\n>>> RECOMMENDATION: {fastest.model_name} (best speed AND quality)"
            )
        else:
            # Trade-off
            print(
                f"\n>>> TRADE-OFF: {fastest.model_name} for speed, {best_quality.model_name} for quality"
            )


def main():
    print("Benchmarking embedding models for scientific paper retrieval...")
    print(f"Test queries: {len(TEST_QUERIES)}")
    print(f"Test documents: {len(TEST_DOCUMENTS)}")

    results = []

    # FastEmbed (always available)
    results.append(benchmark_fastembed())

    # Sentence-transformers variants
    results.append(benchmark_sentence_transformers("all-MiniLM-L6-v2"))
    results.append(benchmark_sentence_transformers("all-mpnet-base-v2"))

    # SPECTER2 (scientific)
    results.append(benchmark_specter2())

    print_summary(results)


if __name__ == "__main__":
    main()
