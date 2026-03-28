"""Report generation — markdown, JSON, and terminal dashboard."""

from src.evaluation.report.json_export import JSONExporter
from src.evaluation.report.markdown import MarkdownReporter
from src.evaluation.report.terminal import TerminalDashboard

__all__ = ["JSONExporter", "MarkdownReporter", "TerminalDashboard"]
