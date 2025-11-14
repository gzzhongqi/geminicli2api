"""
Configuration constants for the Geminicli2api proxy server.
Centralizes all configuration to avoid duplication across modules.
"""
import os

# API Endpoints
CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"

# Client Configuration
CLI_VERSION = "0.1.5"  # Match current gemini-cli version

# OAuth Configuration
CLIENT_ID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
CLIENT_SECRET = "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"
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

# File Paths
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIAL_FILE = os.path.join(SCRIPT_DIR, os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "oauth_creds.json"))

# Authentication
GEMINI_AUTH_PASSWORD = os.getenv("GEMINI_AUTH_PASSWORD", "123456")

# Default Safety Settings for Google API
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_IMAGE_HATE", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_UNSPECIFIED", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_JAILBREAK", "threshold": "BLOCK_NONE"}
]

# Base Models (without search variants)
BASE_MODELS = [
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
        "topK": 64
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
        "topK": 64
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
        "topK": 64
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
        "topK": 64
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
        "topK": 64
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
        "topK": 64
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
        "topK": 64
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
        "topK": 64
    },
    {
        "name": "models/gemini-3-pro-preview-11-2025",
        "version": "001",
        "displayName": "Gemini 3.0 Pro Preview 11-2025",
        "description": "Preview version of Gemini 3.0 Pro from November 2025",
        "inputTokenLimit": 1048576,
        "outputTokenLimit": 65535,
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
        "temperature": 1.0,
        "maxTemperature": 2.0,
        "topP": 0.95,
        "topK": 64
    }
]

# Generate search variants for applicable models
def _generate_search_variants():
    """Generate search variants for models that support content generation."""
    search_models = []
    base_model_with_variance = [model for model in BASE_MODELS if "gemini-2.5-flash-image" not in model["name"]]
    for model in base_model_with_variance:
        # Only add search variants for models that support content generation
        if "generateContent" in model["supportedGenerationMethods"]:
            search_variant = model.copy()
            search_variant["name"] = model["name"] + "-search"
            search_variant["displayName"] = model["displayName"] + " with Google Search"
            search_variant["description"] = model["description"] + " (includes Google Search grounding)"
            search_models.append(search_variant)
    return search_models

# Generate thinking variants for applicable models
def _generate_thinking_variants():
    """Generate nothinking and maxthinking variants for models that support thinking."""
    thinking_models = []
    base_model_with_variance = [model for model in BASE_MODELS if "gemini-2.5-flash-image" not in model["name"]]
    for model in base_model_with_variance:
        # Only add thinking variants for models that support content generation
        # and contain "gemini-2.5-flash" or "gemini-2.5-pro" in their name
        if ("generateContent" in model["supportedGenerationMethods"] and
            ("gemini-2.5-flash" in model["name"] or "gemini-2.5-pro" in model["name"])):
            
            # Add -nothinking variant
            nothinking_variant = model.copy()
            nothinking_variant["name"] = model["name"] + "-nothinking"
            nothinking_variant["displayName"] = model["displayName"] + " (No Thinking)"
            nothinking_variant["description"] = model["description"] + " (thinking disabled)"
            thinking_models.append(nothinking_variant)
            
            # Add -maxthinking variant
            maxthinking_variant = model.copy()
            maxthinking_variant["name"] = model["name"] + "-maxthinking"
            maxthinking_variant["displayName"] = model["displayName"] + " (Max Thinking)"
            maxthinking_variant["description"] = model["description"] + " (maximum thinking budget)"
            thinking_models.append(maxthinking_variant)
    return thinking_models

# Generate combined variants (search + thinking combinations)
def _generate_combined_variants():
    """Generate combined search and thinking variants."""
    combined_models = []
    for model in BASE_MODELS:
        # Only add combined variants for models that support content generation
        # and contain "gemini-2.5-flash" or "gemini-2.5-pro" in their name
        if ("generateContent" in model["supportedGenerationMethods"] and
            ("gemini-2.5-flash" in model["name"] or "gemini-2.5-pro" in model["name"])):
            
            # search + nothinking
            search_nothinking = model.copy()
            search_nothinking["name"] = model["name"] + "-search-nothinking"
            search_nothinking["displayName"] = model["displayName"] + " with Google Search (No Thinking)"
            search_nothinking["description"] = model["description"] + " (includes Google Search grounding, thinking disabled)"
            combined_models.append(search_nothinking)
            
            # search + maxthinking
            search_maxthinking = model.copy()
            search_maxthinking["name"] = model["name"] + "-search-maxthinking"
            search_maxthinking["displayName"] = model["displayName"] + " with Google Search (Max Thinking)"
            search_maxthinking["description"] = model["description"] + " (includes Google Search grounding, maximum thinking budget)"
            combined_models.append(search_maxthinking)
    return combined_models

# Supported Models (includes base models, search variants, and thinking variants)
# Combine all models and then sort them by name to group variants together
all_models = BASE_MODELS + _generate_search_variants() + _generate_thinking_variants()
SUPPORTED_MODELS = sorted(all_models, key=lambda x: x['name'])

# Helper function to get base model name from any variant
def get_base_model_name(model_name):
    """Convert variant model name to base model name."""
    # Remove all possible suffixes in order
    suffixes = ["-maxthinking", "-nothinking", "-search"]
    for suffix in suffixes:
        if model_name.endswith(suffix):
            return model_name[:-len(suffix)]
    return model_name

# Helper function to check if model uses search grounding
def is_search_model(model_name):
    """Check if model name indicates search grounding should be enabled."""
    return "-search" in model_name

# Helper function to check if model uses no thinking
def is_nothinking_model(model_name):
    """Check if model name indicates thinking should be disabled."""
    return "-nothinking" in model_name

# Helper function to check if model uses max thinking
def is_maxthinking_model(model_name):
    """Check if model name indicates maximum thinking budget should be used."""
    return "-maxthinking" in model_name

# Helper function to get thinking budget for a model
def get_thinking_budget(model_name):
    """Get the appropriate thinking budget for a model based on its name and variant."""
    base_model = get_base_model_name(model_name)
    
    if is_nothinking_model(model_name):
        if "gemini-2.5-flash" in base_model:
            return 0  # No thinking for flash
        elif "gemini-2.5-pro" in base_model:
            return 128  # Limited thinking for pro
        elif "gemini-3-pro" in base_model:
            return 128  # Limited thinking for pro
    elif is_maxthinking_model(model_name):
        if "gemini-2.5-flash" in base_model:
            return 24576
        elif "gemini-2.5-pro" in base_model:
            return 32768
        elif "gemini-3-pro" in base_model:
            return 45000
    else:
        # Default thinking budget for regular models
        return -1  # Default for all models

# Helper function to check if thinking should be included in output
def should_include_thoughts(model_name):
    """Check if thoughts should be included in the response."""
    if is_nothinking_model(model_name):
        # For nothinking mode, still include thoughts if it's a pro model
        base_model = get_base_model_name(model_name)
        return "gemini-2.5-pro" in base_model or "gemini-3-pro" in base_model
    else:
        # For all other modes, include thoughts
        return True