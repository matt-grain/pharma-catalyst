"""
CrewAI crew definition for pharma-agents.

This module defines the agent crew that autonomously iterates on
molecular property prediction models.
"""

import os
import re
import time

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from .tools import (
    CodeCheckTool,
    CompoundLookupTool,
    EditTrainPyTool,
    ExperimentalValidationTool,
    FetchMorePapersTool,
    InstallPackageTool,
    KnowledgeQueryTool,
    LiteratureQueryTool,
    PubMedSearchTool,
    ReadTrainPyTool,
    RemovePaperTool,
    RunTrainPyTool,
    SearchAndStoreTool,
    SearchTrainPyTool,
    SkillDiscoveryTool,
    SkillLoaderTool,
    ToolUniverseSearchTool,
    WriteTrainPyTool,
)


class HypothesisOutput(BaseModel):
    """Structured output from hypothesis agent."""

    proposal: str
    reasoning: str
    change_description: str
    literature_insight: str | None = None


class LoggingLLM(LLM):
    """LLM wrapper that logs call duration and enforces free-tier rate limits."""

    _call_count: int = 0
    _last_call_time: float = 0.0
    # Free tier: minimum gap between calls (seconds)
    _FREE_TIER_MIN_GAP: float = 4.0
    # Max retries on rate limit (429) errors
    _RATE_LIMIT_MAX_RETRIES: int = 3

    @staticmethod
    def _parse_retry_delay(error_msg: str) -> float | None:
        """Extract retryDelay from API error message (e.g. 'retryDelay': '31s')."""
        match = re.search(r"retry\s*[iI]n\s+(\d+\.?\d*)", error_msg)
        if match:
            return float(match.group(1))
        match = re.search(r"retryDelay['\"]:\s*['\"](\d+)", error_msg)
        if match:
            return float(match.group(1))
        return None

    def call(self, *args, **kwargs):
        # Enforce per-call rate limit on free tier
        if is_free_tier():
            now = time.perf_counter()
            elapsed = now - LoggingLLM._last_call_time
            if elapsed < self._FREE_TIER_MIN_GAP:
                wait = self._FREE_TIER_MIN_GAP - elapsed
                logger.debug(f"Free tier throttle: waiting {wait:.1f}s")
                time.sleep(wait)
            LoggingLLM._last_call_time = time.perf_counter()

        LoggingLLM._call_count += 1
        call_id = LoggingLLM._call_count
        logger.info(f"LLM CALL #{call_id} START | {self.model}")
        print(f"[LLM #{call_id}] Calling {self.model}...")
        start = time.perf_counter()

        last_error = None
        for attempt in range(1, self._RATE_LIMIT_MAX_RETRIES + 1):
            try:
                result = super().call(*args, **kwargs)
                duration = time.perf_counter() - start
                logger.info(f"LLM CALL #{call_id} OK | {duration:.2f}s")
                print(f"[LLM #{call_id}] Done in {duration:.2f}s")
                LoggingLLM._last_call_time = time.perf_counter()
                return result
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                duration = time.perf_counter() - start

                # Only retry on rate limit errors
                if "429" not in error_msg and "rate limit" not in error_msg:
                    logger.error(f"LLM CALL #{call_id} FAIL | {duration:.2f}s | {e}")
                    print(f"[LLM #{call_id}] FAILED after {duration:.2f}s: {e}")
                    raise

                # Parse API-suggested delay or use default backoff
                delay = self._parse_retry_delay(str(e)) or (30 * attempt)
                delay += 5  # safety buffer
                logger.warning(
                    f"LLM CALL #{call_id} rate limited (attempt {attempt}/"
                    f"{self._RATE_LIMIT_MAX_RETRIES}) — waiting {delay:.0f}s"
                )
                print(f"[LLM #{call_id}] Rate limited — waiting {delay:.0f}s...")
                time.sleep(delay)
                LoggingLLM._last_call_time = time.perf_counter()

        # All retries exhausted
        logger.error(
            f"LLM CALL #{call_id} FAIL after {self._RATE_LIMIT_MAX_RETRIES} retries | {last_error}"
        )
        raise last_error  # type: ignore[misc]


# Load .env file
load_dotenv()


def is_free_tier() -> bool:
    """Check if running on Gemini free tier (needs rate throttling).

    Only applies to Gemini models — other providers (Groq, etc.) have
    generous free tiers that don't need throttling.
    """
    model = os.getenv("LLM_MODEL", "")
    if not model.startswith("gemini"):
        return False
    return os.getenv("GEMINI_TIER", "paid").lower() == "free"


def get_max_rpm() -> int:
    """Get max requests per minute based on API tier."""
    return 10 if is_free_tier() else 30


# Provider → env var mapping for API keys
_PROVIDER_KEY_MAP: dict[str, list[str]] = {
    "gemini": ["GOOGLE_API_KEY"],
    "groq": ["GROQ_API_KEY", "GROQ_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "cerebras": ["CEREBRAS_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
}


def get_provider(model: str) -> str:
    """Extract provider prefix from model string (e.g. 'groq/llama...' → 'groq')."""
    return model.split("/")[0] if "/" in model else "openai"


def get_api_key(model: str) -> str:
    """Get API key for the model's provider from environment variables."""
    provider = get_provider(model)
    env_vars = _PROVIDER_KEY_MAP.get(provider, [f"{provider.upper()}_API_KEY"])
    for var in env_vars:
        key = os.getenv(var)
        if key:
            return key
    raise EnvironmentError(
        f"No API key found for provider '{provider}'. Set one of: {', '.join(env_vars)}"
    )


def get_llm(temperature: float = 0.7) -> LLM:
    """Get configured LLM from environment variables (provider-agnostic)."""
    model = os.getenv("LLM_MODEL")
    if not model:
        raise EnvironmentError(
            "LLM_MODEL environment variable is required. "
            "Example: LLM_MODEL=gemini/gemini-2.0-flash or groq/llama-3.1-8b-instant"
        )
    api_key = get_api_key(model)
    return LoggingLLM(model=model, api_key=api_key, temperature=temperature)


@CrewBase
class PharmaAgentsCrew:
    """
    Autonomous multi-agent crew for molecular ML optimization.

    The crew follows a sequential process:
    1. Hypothesis Agent proposes a change
    2. Model Agent implements the change in train.py
    3. Evaluator Agent runs training and reports results
    """

    agents_config = "agents.yaml"
    tasks_config = "tasks.yaml"

    @agent
    def archivist_agent(self) -> Agent:
        """Literature Archivist who gathers recent research."""
        return Agent(
            config=self.agents_config["archivist_agent"],  # type: ignore[index]
            llm=get_llm(),
            tools=[
                SearchAndStoreTool(),
                PubMedSearchTool(),
                RemovePaperTool(),
            ],
            max_iter=20,
            max_execution_time=600,  # 10 min — network I/O heavy
            verbose=True,
        )

    @agent
    def hypothesis_agent(self) -> Agent:
        """Research Scientist who proposes improvements."""
        return Agent(
            config=self.agents_config["hypothesis_agent"],  # type: ignore[index]
            llm=get_llm(temperature=0.8),  # Higher for creativity
            tools=[
                ReadTrainPyTool(),
                LiteratureQueryTool(),
                KnowledgeQueryTool(),  # Internal reports, assay data, SOPs
                SkillDiscoveryTool(),
                SkillLoaderTool(),
                FetchMorePapersTool(),  # Request fresh papers when stuck
                CompoundLookupTool(),
                ToolUniverseSearchTool(),
            ],
            max_iter=15,
            max_execution_time=300,  # 5 min — mostly LLM reasoning
            verbose=True,
        )

    @agent
    def model_agent(self) -> Agent:
        """ML Engineer who implements changes."""
        return Agent(
            config=self.agents_config["model_agent"],  # type: ignore[index]
            llm=get_llm(temperature=0.3),  # Lower for code generation
            tools=[
                ReadTrainPyTool(),
                SearchTrainPyTool(),
                WriteTrainPyTool(),
                EditTrainPyTool(),
                CodeCheckTool(),
                InstallPackageTool(),
            ],
            max_iter=40,
            max_execution_time=600,  # 10 min — code fix cycles + training
            verbose=True,
        )

    @agent
    def evaluator_agent(self) -> Agent:
        """QA Scientist who evaluates results."""
        return Agent(
            config=self.agents_config["evaluator_agent"],
            llm=get_llm(),
            tools=[
                ReadTrainPyTool(),
                RunTrainPyTool(),
                ExperimentalValidationTool(),
            ],
            max_iter=10,
            max_execution_time=300,  # 5 min — includes training timeout
            verbose=True,
        )

    @task
    def archivist_task(self) -> Task:
        """Task: Gather recent research papers (runs async in parallel)."""
        return Task(
            config=self.tasks_config["archivist_task"],  # type: ignore[index,call-arg]
            async_execution=True,  # Run in parallel with other tasks
        )

    @task
    def hypothesis_task(self) -> Task:
        """Task: Propose an improvement."""
        return Task(
            config=self.tasks_config["hypothesis_task"],  # type: ignore[index,call-arg]
            output_pydantic=HypothesisOutput,
        )

    @task
    def implement_task(self) -> Task:
        """Task: Implement the proposed change."""
        return Task(
            config=self.tasks_config["implement_task"],  # type: ignore[index,call-arg]
            context=[self.hypothesis_task()],  # type: ignore[list-item]
        )

    @task
    def evaluate_task(self) -> Task:
        """Task: Run and evaluate the change."""
        return Task(
            config=self.tasks_config["evaluate_task"],
            context=[
                self.implement_task()
            ],  # Know what was changed  # type: ignore[list-item]
        )

    @crew
    def crew(self) -> Crew:
        """Create the pharma-agents crew (without archivist)."""
        return Crew(
            agents=[
                self.hypothesis_agent(),
                self.model_agent(),
                self.evaluator_agent(),
            ],
            tasks=[
                self.hypothesis_task(),
                self.implement_task(),
                self.evaluate_task(),
            ],
            process=Process.sequential,
            max_rpm=get_max_rpm(),
            verbose=True,
        )

    def archivist_crew(self) -> Crew:
        """Create archivist-only crew for standalone literature research."""
        return Crew(
            agents=[self.archivist_agent()],
            tasks=[
                Task(
                    config=self.tasks_config["archivist_task"],  # type: ignore[index,call-arg]
                ),
            ],
            process=Process.sequential,
            max_rpm=get_max_rpm(),
            verbose=True,
        )

    def crew_with_archivist(self) -> Crew:
        """Create crew with archivist running async in parallel.

        The archivist_task has async_execution=True, so it runs in background
        while hypothesis_task starts immediately. Literature will be ready
        for subsequent runs.
        """
        return Crew(
            agents=[
                self.archivist_agent(),
                self.hypothesis_agent(),
                self.model_agent(),
                self.evaluator_agent(),
            ],
            tasks=[
                self.archivist_task(),  # async - runs in parallel
                self.hypothesis_task(),
                self.implement_task(),
                self.evaluate_task(),
            ],
            process=Process.sequential,
            max_rpm=get_max_rpm(),
            verbose=True,
        )

    def hypothesis_crew(self) -> Crew:
        """Hypothesis-only crew for review panel flow."""
        return Crew(
            agents=[self.hypothesis_agent()],
            tasks=[self.hypothesis_task()],
            process=Process.sequential,
            max_rpm=get_max_rpm(),
            verbose=True,
        )

    def implementation_crew(self) -> Crew:
        """Implementation+evaluation crew, receives approved hypothesis via inputs."""
        impl_task = Task(
            config=self.tasks_config["implement_task"],  # type: ignore[index,call-arg]
        )
        eval_task = Task(
            config=self.tasks_config["evaluate_task"],
            context=[impl_task],  # type: ignore[list-item]
        )
        return Crew(
            agents=[self.model_agent(), self.evaluator_agent()],
            tasks=[impl_task, eval_task],
            process=Process.sequential,
            max_rpm=get_max_rpm(),
            verbose=True,
        )
