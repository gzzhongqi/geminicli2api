"""
Gemini model definitions and variant generators.
"""

from typing import Any, Dict, List

# Base Models (without search variants)
BASE_MODELS: List[Dict[str, Any]] = [
    {
        "name": "models/gemini-2.5-pro-preview-03-25",
        "version": "001",
        "displayName": "Gemini 2.5 Pro Preview 03-25",
        "description": "Preview version of Gemini 2.5 Pro from May 6th",
        "inputTokenLimit": 1048576,
        "outputTokenLimit": 65535,
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
        "temperature": 1.0,
        "maxTemperature": 2.0,
        "topP": 0.95,
        "topK": 64,
    },
    {
        "name": "models/gemini-2.5-pro-preview-05-06",
        "version": "001",
        "displayName": "Gemini 2.5 Pro Preview 05-06",
        "description": "Preview version of Gemini 2.5 Pro from May 6th",
        "inputTokenLimit": 1048576,
        "outputTokenLimit": 65535,
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
        "temperature": 1.0,
        "maxTemperature": 2.0,
        "topP": 0.95,
        "topK": 64,
    },
    {
        "name": "models/gemini-2.5-pro-preview-06-05",
        "version": "001",
        "displayName": "Gemini 2.5 Pro Preview 06-05",
        "description": "Preview version of Gemini 2.5 Pro from June 5th",
        "inputTokenLimit": 1048576,
        "outputTokenLimit": 65535,
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
        "temperature": 1.0,
        "maxTemperature": 2.0,
        "topP": 0.95,
        "topK": 64,
    },
    {
        "name": "models/gemini-2.5-pro",
        "version": "001",
        "displayName": "Gemini 2.5 Pro",
        "description": "Advanced multimodal model with enhanced capabilities",
        "inputTokenLimit": 1048576,
        "outputTokenLimit": 65535,
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
        "temperature": 1.0,
        "maxTemperature": 2.0,
        "topP": 0.95,
        "topK": 64,
    },
    {
        "name": "models/gemini-2.5-flash-preview-05-20",
        "version": "001",
        "displayName": "Gemini 2.5 Flash Preview 05-20",
        "description": "Preview version of Gemini 2.5 Flash from May 20th",
        "inputTokenLimit": 1048576,
        "outputTokenLimit": 65535,
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
        "temperature": 1.0,
        "maxTemperature": 2.0,
        "topP": 0.95,
        "topK": 64,
    },
    {
        "name": "models/gemini-2.5-flash-preview-04-17",
        "version": "001",
        "displayName": "Gemini 2.5 Flash Preview 04-17",
        "description": "Preview version of Gemini 2.5 Flash from April 17th",
        "inputTokenLimit": 1048576,
        "outputTokenLimit": 65535,
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
        "temperature": 1.0,
        "maxTemperature": 2.0,
        "topP": 0.95,
        "topK": 64,
    },
    {
        "name": "models/gemini-2.5-flash",
        "version": "001",
        "displayName": "Gemini 2.5 Flash",
        "description": "Fast and efficient multimodal model with latest improvements",
        "inputTokenLimit": 1048576,
        "outputTokenLimit": 65535,
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
        "temperature": 1.0,
        "maxTemperature": 2.0,
        "topP": 0.95,
        "topK": 64,
    },
    {
        "name": "models/gemini-2.5-flash-image-preview",
        "version": "001",
        "displayName": "Gemini 2.5 Flash Image Preview",
        "description": "Gemini 2.5 Flash Image Preview",
        "inputTokenLimit": 32768,
        "outputTokenLimit": 32768,
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
        "temperature": 1.0,
        "maxTemperature": 2.0,
        "topP": 0.95,
        "topK": 64,
    },
    {
        "name": "models/gemini-3-pro-preview",
        "version": "001",
        "displayName": "Gemini 3.0 Pro Preview 11-2025",
        "description": "Preview version of Gemini 3.0 Pro from November 2025",
        "inputTokenLimit": 1048576,
        "outputTokenLimit": 65535,
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
        "temperature": 1.0,
        "maxTemperature": 2.0,
        "topP": 0.95,
        "topK": 64,
    },
]


def _generate_search_variants() -> List[Dict[str, Any]]:
    """Generate search variants for models that support content generation."""
    search_models = []
    base_model_with_variance = [
        model for model in BASE_MODELS if "gemini-2.5-flash-image" not in model["name"]
    ]
    for model in base_model_with_variance:
        if "generateContent" in model["supportedGenerationMethods"]:
            search_variant = model.copy()
            search_variant["name"] = model["name"] + "-search"
            search_variant["displayName"] = model["displayName"] + " with Google Search"
            search_variant["description"] = (
                model["description"] + " (includes Google Search grounding)"
            )
            search_models.append(search_variant)
    return search_models


def _generate_thinking_variants() -> List[Dict[str, Any]]:
    """Generate nothinking and maxthinking variants for models that support thinking."""
    thinking_models = []
    base_model_with_variance = [
        model for model in BASE_MODELS if "gemini-2.5-flash-image" not in model["name"]
    ]
    for model in base_model_with_variance:
        if "generateContent" in model["supportedGenerationMethods"] and (
            "gemini-2.5-flash" in model["name"] or "gemini-2.5-pro" in model["name"]
        ):
            # Add -nothinking variant
            nothinking_variant = model.copy()
            nothinking_variant["name"] = model["name"] + "-nothinking"
            nothinking_variant["displayName"] = model["displayName"] + " (No Thinking)"
            nothinking_variant["description"] = (
                model["description"] + " (thinking disabled)"
            )
            thinking_models.append(nothinking_variant)

            # Add -maxthinking variant
            maxthinking_variant = model.copy()
            maxthinking_variant["name"] = model["name"] + "-maxthinking"
            maxthinking_variant["displayName"] = (
                model["displayName"] + " (Max Thinking)"
            )
            maxthinking_variant["description"] = (
                model["description"] + " (maximum thinking budget)"
            )
            thinking_models.append(maxthinking_variant)
    return thinking_models


# Supported Models (includes base models, search variants, and thinking variants)
# Combine all models and then sort them by name to group variants together
_all_models = BASE_MODELS + _generate_search_variants() + _generate_thinking_variants()
SUPPORTED_MODELS: List[Dict[str, Any]] = sorted(_all_models, key=lambda x: x["name"])
