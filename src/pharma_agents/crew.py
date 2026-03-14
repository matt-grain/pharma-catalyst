"""
CrewAI crew definition for pharma-agents.

This module defines the agent crew that autonomously iterates on
molecular property prediction models.
"""

import time
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from loguru import logger
import os

from .tools.custom_tools import (
    ReadTrainPyTool,
    WriteTrainPyTool,
    RunTrainPyTool,
    CodeCheckTool,
    AlphaxivTool,
    ArxivSearchTool,
    LiteratureStoreTool,
    LiteratureQueryTool,
    SkillLoaderTool,
    FetchMorePapersTool,
)


class LoggingLLM(LLM):
    """LLM wrapper that logs call duration."""

    _call_count: int = 0

    def call(self, *args, **kwargs):
        LoggingLLM._call_count += 1
        call_id = LoggingLLM._call_count
        logger.info(f"LLM CALL #{call_id} START | {self.model}")
        print(f"[LLM #{call_id}] Calling {self.model}...")
        start = time.perf_counter()
        try:
            result = super().call(*args, **kwargs)
            duration = time.perf_counter() - start
            logger.info(f"LLM CALL #{call_id} OK | {duration:.2f}s")
            print(f"[LLM #{call_id}] Done in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.perf_counter() - start
            logger.error(f"LLM CALL #{call_id} FAIL | {duration:.2f}s | {e}")
            print(f"[LLM #{call_id}] FAILED after {duration:.2f}s: {e}")
            raise


# Load .env file
load_dotenv()


# Configure LLM from environment
def get_llm() -> LLM:
    """Get configured LLM from environment variables."""
    model = os.getenv("LLM_MODEL", "gemini/gemini-3-flash-preview")
    return LoggingLLM(
        model=model,
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )


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
                ArxivSearchTool(),
                AlphaxivTool(),
                LiteratureStoreTool(),
            ],
            max_iter=30,  # Allow more iterations for research
            verbose=True,
        )

    @agent
    def hypothesis_agent(self) -> Agent:
        """Research Scientist who proposes improvements."""
        return Agent(
            config=self.agents_config["hypothesis_agent"],  # type: ignore[index]
            llm=get_llm(),
            tools=[
                ReadTrainPyTool(),
                LiteratureQueryTool(),
                SkillLoaderTool(),
                FetchMorePapersTool(),  # Request fresh papers when stuck
            ],
            verbose=True,
        )

    @agent
    def model_agent(self) -> Agent:
        """ML Engineer who implements changes."""
        return Agent(
            config=self.agents_config["model_agent"],  # type: ignore[index]
            llm=get_llm(),
            tools=[
                ReadTrainPyTool(),
                WriteTrainPyTool(),
                CodeCheckTool(),
            ],
            max_iter=40,  # Allow more iterations for code fix cycles
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
            ],
            verbose=True,
        )

    @task
    def archivist_task(self) -> Task:
        """Task: Gather recent research papers."""
        return Task(
            config=self.tasks_config["archivist_task"],  # type: ignore[index,call-arg]
        )

    @task
    def hypothesis_task(self) -> Task:
        """Task: Propose an improvement."""
        return Task(
            config=self.tasks_config["hypothesis_task"],  # type: ignore[index,call-arg]
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
        """Task: Run and evaluate the change (no context - just run and compare)."""
        return Task(
            config=self.tasks_config["evaluate_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Create the pharma-agents crew (without archivist - use research_crew for that)."""
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
            verbose=True,
        )

    def research_crew(self) -> Crew:
        """Create a research-only crew (archivist only) for literature gathering.

        Note: Not using @crew decorator - this is a standalone crew for literature research.
        """
        return Crew(
            agents=[self.archivist_agent()],
            tasks=[self.archivist_task()],
            process=Process.sequential,
            verbose=True,
        )
