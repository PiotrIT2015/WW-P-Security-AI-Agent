"""
agent/exceptions.py

Exceptions used by WW-P Security AI Agent.
"""


class AgentError(Exception):
    """Base exception for Agent Core."""

    pass


class AgentConfigurationError(AgentError):
    """Raised when Agent Core configuration is invalid."""

    pass


class AgentStateError(AgentError):
    """Raised when an invalid state operation is attempted."""

    pass


class AgentExecutionError(AgentError):
    """Raised when agent execution fails."""

    pass


class AgentIterationLimitError(AgentExecutionError):
    """Raised when the agent exceeds its iteration limit."""

    pass