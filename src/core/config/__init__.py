"""
src.core.config
===============
Public API for the configuration layer.

Import from here — never import directly from ``settings.py``.

    from src.core.config import settings                  # singleton
    from src.core.config import get_settings              # factory
    from src.core.config import get_settings_cached       # FastAPI dependency
    from src.core.config import reset_settings            # test helper
    from src.core.config import Settings                  # type hint
    from src.core.config import (
        AppEnvironment, LLMProvider, EmbeddingProvider,
        LogLevel, LogFormat, CacheBackend, StreamlitTheme,
    )
"""

from core.config.settings import (
    # Enumerations
    AppEnvironment,
    CacheBackend,
    EmbeddingProvider,
    LLMProvider,
    LogFormat,
    LogLevel,
    StreamlitTheme,
    # Nested settings models (for type hints in other modules)
    AnthropicSettings,
    APISettings,
    AppSettings,
    AuthSettings,
    CacheSettings,
    DirectorySettings,
    EmbeddingSettings,
    GeminiSettings,
    GroqSettings,
    IngestionSettings,
    LangSmithSettings,
    LoggingSettings,
    OllamaSettings,
    OpenAISettings,
    QdrantSettings,
    RateLimitSettings,
    RerankerSettings,
    RetrievalSettings,
    StreamlitSettings,
    # Root settings class
    Settings,
    # Factory / helpers
    get_settings,
    get_settings_cached,
    reset_settings,
    # Convenience singleton
    settings,
)

__all__: list[str] = [
    # Enumerations
    "AppEnvironment",
    "CacheBackend",
    "EmbeddingProvider",
    "LLMProvider",
    "LogFormat",
    "LogLevel",
    "StreamlitTheme",
    # Nested settings models
    "AnthropicSettings",
    "APISettings",
    "AppSettings",
    "AuthSettings",
    "CacheSettings",
    "DirectorySettings",
    "EmbeddingSettings",
    "GeminiSettings",
    "GroqSettings",
    "IngestionSettings",
    "LangSmithSettings",
    "LoggingSettings",
    "OllamaSettings",
    "OpenAISettings",
    "QdrantSettings",
    "RateLimitSettings",
    "RerankerSettings",
    "RetrievalSettings",
    "StreamlitSettings",
    # Root
    "Settings",
    # Factory / helpers
    "get_settings",
    "get_settings_cached",
    "reset_settings",
    # Singleton
    "settings",
]
