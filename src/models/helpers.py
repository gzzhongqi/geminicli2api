"""
Helper functions for working with Gemini models.
"""


def get_base_model_name(model_name: str) -> str:
    """Convert variant model name to base model name."""
    suffixes = ["-maxthinking", "-nothinking", "-search"]
    for suffix in suffixes:
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
    return model_name


def is_search_model(model_name: str) -> bool:
    """Check if model name indicates search grounding should be enabled."""
    return "-search" in model_name


def is_nothinking_model(model_name: str) -> bool:
    """Check if model name indicates thinking should be disabled."""
    return "-nothinking" in model_name


def is_maxthinking_model(model_name: str) -> bool:
    """Check if model name indicates maximum thinking budget should be used."""
    return "-maxthinking" in model_name


def get_thinking_budget(model_name: str) -> int:
    """Get the appropriate thinking budget for a model based on its name and variant."""
    base_model = get_base_model_name(model_name)

    if is_nothinking_model(model_name):
        if "gemini-2.5-flash" in base_model:
            return 0  # No thinking for flash
        elif "gemini-2.5-pro" in base_model:
            return 128  # Limited thinking for pro
        elif "gemini-3-pro" in base_model:
            return 128  # Limited thinking for pro
        return 0  # Default for nothinking
    elif is_maxthinking_model(model_name):
        if "gemini-2.5-flash" in base_model:
            return 24576
        elif "gemini-2.5-pro" in base_model:
            return 32768
        elif "gemini-3-pro" in base_model:
            return 45000
        return 32768  # Default for maxthinking

    # Default thinking budget for regular models
    return -1


def should_include_thoughts(model_name: str) -> bool:
    """Check if thoughts should be included in the response."""
    if is_nothinking_model(model_name):
        # For nothinking mode, still include thoughts if it's a pro model
        base_model = get_base_model_name(model_name)
        return "gemini-2.5-pro" in base_model or "gemini-3-pro" in base_model
    else:
        # For all other modes, include thoughts
        return True
