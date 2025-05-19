import logging
from setup.config import config

logger = logging.getLogger(__name__)

def get_llm():
    """Get a configured LLM based on settings."""
    provider = config.DEFAULT_LLM_PROVIDER.lower()
    match provider:
        case "gemini":
            return get_gemini_model()
        case "openai":
            return get_openai_model()
        case "anthropic":
            return get_anthropic_model()
        case "ollama":
            return get_ollama_model()
        case _:
            logger.warning(f"Unknown provider '{provider}', falling back to Gemini")
            return get_gemini_model()

def get_gemini_model():
    """Get Google model."""
    try:
        # (import only if we are using it)
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Error loading Gemini model: {e}", exc_info=True)
        raise

def get_openai_model():
    """Get OpenAI model."""
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Error loading OpenAI model: {e}", exc_info=True)
        raise

def get_anthropic_model():
    """Get Anthropic model."""
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Error loading Anthropic model: {e}", exc_info=True)
        raise

def get_ollama_model():
    """Get Ollama model."""
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_HOST,
            temperatur=0.7,
        )
    except Exception as e:
        logger.error(f"Error loading Ollama model: {e}", exc_info=True)
        raise
    
def get_embedding_model():
    """Get embedding model for memory vectorization."""
    provider = config.DEFAULT_LLM_PROVIDER.lower()
    
    if provider == "openai" and config.OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        # Use OpenAI's embeddings if we have an API key
        return OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    elif provider == "google" and config.GEMINI_API_KEY:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        # Use Google's embeddings if we have an API key
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=config.GEMINI_API_KEY,
        )
    else:
        # Fallback - Huggingface's Embeddings (no API key required!)
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
