"""
CrewAI crew definition for pharma-agents.

This module defines the agent crew that autonomously iterates on
molecular property prediction models.
"""

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool, FileWriterTool
from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()


# Configure Gemini as the LLM
def get_llm() -> LLM:
    """Get configured LLM (Gemini by default)."""
    return LLM(
        model="gemini/gemini-3-flash-preview",
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

    def __init__(self):
        self.tools_dir = Path(__file__).parent / "tools"

    @agent
    def hypothesis_agent(self) -> Agent:
        """Research Scientist who proposes improvements."""
        return Agent(
            config=self.agents_config["hypothesis_agent"],
            llm=get_llm(),
            tools=[
                FileReadTool(file_path=str(self.tools_dir / "train.py")),
            ],
            verbose=True,
        )

    @agent
    def model_agent(self) -> Agent:
        """ML Engineer who implements changes."""
        return Agent(
            config=self.agents_config["model_agent"],
            llm=get_llm(),
            tools=[
                FileReadTool(file_path=str(self.tools_dir / "train.py")),
                FileWriterTool(),
            ],
            verbose=True,
        )

    @agent
    def evaluator_agent(self) -> Agent:
        """QA Scientist who evaluates results."""
        # Note: In production, this would use a custom tool to run train.py
        # For now, we simulate with file reading
        return Agent(
            config=self.agents_config["evaluator_agent"],
            llm=get_llm(),
            tools=[
                FileReadTool(file_path=str(self.tools_dir / "train.py")),
            ],
            verbose=True,
        )

    @task
    def hypothesis_task(self) -> Task:
        """Task: Propose an improvement."""
        return Task(
            config=self.tasks_config["hypothesis_task"],
        )

    @task
    def implement_task(self) -> Task:
        """Task: Implement the proposed change."""
        return Task(
            config=self.tasks_config["implement_task"],
        )

    @task
    def evaluate_task(self) -> Task:
        """Task: Run and evaluate the change."""
        return Task(
            config=self.tasks_config["evaluate_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Create the pharma-agents crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
