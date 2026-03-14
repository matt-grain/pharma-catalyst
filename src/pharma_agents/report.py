"""
HTML report generator with Plotly charts.

Generates visual reports showing experiment progress and metrics.
"""

from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_run_report(
    run_id: int,
    experiment_name: str,
    metric: str,
    direction: str,
    baseline_score: float,
    experiments: list[dict],
    output_dir: Path,
) -> Path:
    """
    Generate an HTML report for a run.

    Args:
        run_id: Run number
        experiment_name: Name of the experiment (bbbp, solubility)
        metric: Metric name (ROC_AUC, RMSE)
        direction: 'higher_is_better' or 'lower_is_better'
        baseline_score: Starting baseline score
        experiments: List of experiment dicts with iteration, score_after, etc.
        output_dir: Directory to save the report

    Returns:
        Path to the generated HTML file
    """
    higher_is_better = direction == "higher_is_better"

    # Extract data for charts - start with baseline at iteration 0
    iterations = [0]
    scores = [baseline_score]
    results = ["baseline"]
    hypotheses = ["Baseline"]

    for exp in experiments:
        iterations.append(exp["iteration"])
        # Support both old (score_after) and new (rmse) field names
        score = exp.get("score_after") or exp.get("rmse") or baseline_score
        scores.append(score)
        # Support both old (result) and new (success) field names
        if "result" in exp:
            results.append(exp["result"])
        else:
            results.append("success" if exp.get("success") else "failure")
        hypotheses.append(exp.get("hypothesis", "")[:50] + "...")

    # Calculate final stats
    successful_exps = [
        e
        for e in experiments
        if e.get("result") == "success" or e.get("success") is True
    ]
    final_score = scores[-1] if scores else baseline_score
    total_improvement = (
        ((final_score - baseline_score) / baseline_score) * 100
        if higher_is_better
        else ((baseline_score - final_score) / baseline_score) * 100
    )

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"{metric} Over Iterations",
            "Before vs After",
            "Success Rate",
            "Improvement by Iteration",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "pie"}, {"type": "bar"}],
        ],
    )

    # Chart 1: Score progression line chart
    colors = [
        "#6C757D" if r == "baseline" else "#28A745" if r == "success" else "#DC3545"
        for r in results
    ]
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=scores,
            mode="lines+markers",
            name=metric,
            line={"color": "#2E86AB", "width": 3},
            marker={
                "size": 12,
                "color": colors,
                "line": {"width": 2, "color": "white"},
            },
            hovertemplate="Iter %{x}<br>" + metric + ": %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add baseline reference line
    fig.add_hline(
        y=baseline_score,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Baseline: {baseline_score:.4f}",
        row=1,
        col=1,
    )

    # Chart 2: Before/After bar chart
    fig.add_trace(
        go.Bar(
            x=["Baseline", "Final"],
            y=[baseline_score, final_score],
            marker_color=["#6C757D", "#28A745" if total_improvement > 0 else "#DC3545"],
            text=[f"{baseline_score:.4f}", f"{final_score:.4f}"],
            textposition="outside",
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Chart 3: Success rate pie chart
    success_count = len(
        [
            e
            for e in experiments
            if e.get("result") == "success" or e.get("success") is True
        ]
    )
    failure_count = len(experiments) - success_count
    fig.add_trace(
        go.Pie(
            labels=["Success", "Failure"],
            values=[success_count, failure_count],
            marker_colors=["#28A745", "#DC3545"],
            hole=0.4,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Chart 4: Improvement per iteration
    improvements = []
    for exp in experiments:
        if (exp.get("result") == "success" or exp.get("success") is True) and exp.get(
            "improvement_pct"
        ):
            improvements.append(exp["improvement_pct"])
        else:
            improvements.append(0)

    fig.add_trace(
        go.Bar(
            x=[f"Iter {i}" for i in iterations],
            y=improvements,
            marker_color=[
                "#28A745" if imp > 0 else "#DC3545" if imp < 0 else "#6C757D"
                for imp in improvements
            ],
            text=[f"{imp:.1f}%" if imp != 0 else "" for imp in improvements],
            textposition="outside",
            hovertemplate="Iter %{x}<br>Improvement: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # Update layout
    arrow = "↑" if higher_is_better else "↓"
    improvement_sign = "+" if total_improvement > 0 else ""

    fig.update_layout(
        title={
            "text": (
                f"<b>{experiment_name.upper()} Run {run_id} Report</b><br>"
                f"<sup>{metric}: {baseline_score:.4f} → {final_score:.4f} "
                f"({improvement_sign}{total_improvement:.1f}% {arrow})</sup>"
            ),
            "x": 0.5,
            "font": {"size": 24},
        },
        showlegend=False,
        height=700,
        template="plotly_white",
        font={"family": "Arial, sans-serif"},
    )

    # Generate experiments table HTML
    table_rows = ""
    for exp in experiments:
        is_success = exp.get("result") == "success" or exp.get("success") is True
        status_badge = (
            '<span style="color: #28A745;">&#10004; SUCCESS</span>'
            if is_success
            else '<span style="color: #DC3545;">&#10008; FAILURE</span>'
        )
        score_after = exp.get("score_after")
        score_str = f"{score_after:.4f}" if score_after else "N/A"
        improvement = exp.get("improvement_pct")
        imp_str = f"{improvement:+.2f}%" if improvement else "N/A"
        hypothesis = exp.get("hypothesis", "N/A")[:80]
        if len(exp.get("hypothesis", "")) > 80:
            hypothesis += "..."

        table_rows += f"""
        <tr>
            <td>{exp["iteration"]}</td>
            <td>{status_badge}</td>
            <td>{score_str}</td>
            <td>{imp_str}</td>
            <td title="{exp.get("hypothesis", "")}">{hypothesis}</td>
            <td>{exp.get("insight", "N/A")}</td>
        </tr>
        """

    # Generate full HTML
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{experiment_name.upper()} Run {run_id} Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .summary-cards {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            min-width: 150px;
        }}
        .card-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2E86AB;
        }}
        .card-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .card.success .card-value {{ color: #28A745; }}
        .card.warning .card-value {{ color: #FFC107; }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .table-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background-color: #2E86AB;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            color: #6c757d;
            margin-top: 30px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 {experiment_name.upper()} Experiment Report</h1>
        <p>Run #{run_id} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    </div>

    <div class="summary-cards">
        <div class="card">
            <div class="card-value">{baseline_score:.4f}</div>
            <div class="card-label">Baseline {metric}</div>
        </div>
        <div class="card success">
            <div class="card-value">{final_score:.4f}</div>
            <div class="card-label">Final {metric}</div>
        </div>
        <div class="card {"success" if total_improvement > 0 else "warning"}">
            <div class="card-value">{improvement_sign}{total_improvement:.1f}%</div>
            <div class="card-label">Improvement</div>
        </div>
        <div class="card">
            <div class="card-value">{len(experiments)}</div>
            <div class="card-label">Experiments</div>
        </div>
        <div class="card success">
            <div class="card-value">{success_count}/{len(experiments)}</div>
            <div class="card-label">Success Rate</div>
        </div>
    </div>

    <div class="chart-container">
        {chart_html}
    </div>

    <div class="table-container">
        <h2>📋 Experiment Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Iter</th>
                    <th>Result</th>
                    <th>{metric}</th>
                    <th>Improvement</th>
                    <th>Hypothesis</th>
                    <th>Insight</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>

    <div class="footer">
        <p>Generated by <strong>pharma-catalyst</strong> 🧬</p>
        <p>Direction: {direction.replace("_", " ")} | Metric: {metric}</p>
    </div>
</body>
</html>
"""

    # Write report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.html"
    report_path.write_text(html_content, encoding="utf-8")

    return report_path


def generate_from_memory(experiment: str, run_id: int | None = None) -> list[Path]:
    """
    Generate reports from memory.json for past runs.

    Args:
        experiment: Experiment name (bbbp, solubility)
        run_id: Specific run ID, or None for all runs

    Returns:
        List of generated report paths
    """
    from .memory import AgentMemory, get_experiments_root, get_baseline_config
    import os

    # Set experiment for config loading
    os.environ["PHARMA_EXPERIMENT"] = experiment

    experiments_dir = get_experiments_root() / experiment
    memory_path = experiments_dir / "memory.json"

    if not memory_path.exists():
        raise FileNotFoundError(f"No memory.json found at {memory_path}")

    # Load memory and config
    memory = AgentMemory(memory_path)
    config = get_baseline_config()
    metric = config["metric"]
    direction = config.get("direction", "lower_is_better")
    baseline_score = config["score"]

    generated = []

    runs_to_process = (
        {run_id: memory.runs[run_id]}
        if run_id and run_id in memory.runs
        else memory.runs
    )

    for rid, run_data in runs_to_process.items():
        if not run_data.experiments:
            continue

        # Convert Experiment dataclass to dict
        experiments = []
        for exp in run_data.experiments:
            experiments.append(
                {
                    "iteration": exp.iteration,
                    "result": exp.result,
                    "score_before": exp.score_before,
                    "score_after": exp.score_after,
                    "improvement_pct": exp.improvement_pct,
                    "hypothesis": exp.hypothesis,
                    "reasoning": exp.reasoning,
                    "insight": exp.insight,
                }
            )

        output_dir = experiments_dir / f"run_{rid:03d}"
        report_path = generate_run_report(
            run_id=rid,
            experiment_name=experiment,
            metric=metric,
            direction=direction,
            baseline_score=baseline_score,
            experiments=experiments,
            output_dir=output_dir,
        )
        generated.append(report_path)

    return generated


if __name__ == "__main__":
    import argparse
    import webbrowser

    parser = argparse.ArgumentParser(
        description="Generate HTML reports from memory.json"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        default="bbbp",
        help="Experiment name (default: bbbp)",
    )
    parser.add_argument(
        "--run",
        "-r",
        type=int,
        default=None,
        help="Specific run ID (default: all runs)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the report in browser after generation",
    )

    args = parser.parse_args()

    print(f"Generating reports for {args.experiment}...")
    try:
        reports = generate_from_memory(args.experiment, args.run)
        for report in reports:
            print(f"  Generated: {report}")

        if args.open and reports:
            webbrowser.open(str(reports[-1]))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
