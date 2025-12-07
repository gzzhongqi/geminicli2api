"""
Helper functions for working with Gemini models.
"""


def get_base_model_name(model_name: str) -> str:
    """Get the base model name."""
    return model_name


def should_include_thoughts(model_name: str) -> bool:
    """Check if thoughts should be included in the response."""
    # Always include thoughts for thinking-capable models
    return True
