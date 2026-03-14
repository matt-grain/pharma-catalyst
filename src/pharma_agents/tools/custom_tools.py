"""Custom CrewAI tools for pharma-agents.

DEPRECATED: Import from pharma_agents.tools instead.
This module re-exports for backwards compatibility.
"""

# Re-export all tools for backwards compatibility
from .arxiv import AlphaxivTool, ArxivSearchTool
from .literature import (
    FetchMorePapersTool,
    LiteratureQueryTool,
    LiteratureStoreTool,
    get_literature_dir,
)
from .skills import SkillLoaderTool
from .training import CodeCheckTool, ReadTrainPyTool, RunTrainPyTool, WriteTrainPyTool

__all__ = [
    "AlphaxivTool",
    "ArxivSearchTool",
    "LiteratureStoreTool",
    "LiteratureQueryTool",
    "FetchMorePapersTool",
    "get_literature_dir",
    "ReadTrainPyTool",
    "WriteTrainPyTool",
    "CodeCheckTool",
    "RunTrainPyTool",
    "SkillLoaderTool",
]
