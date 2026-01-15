import os
import re
import sys
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dotenv import load_dotenv

# Get project root (go up two levels: settings.py -> config -> src -> project_root)
project_root = Path(__file__).resolve().parent.parent.parent

# Type annotations - these will be dynamically assigned during runtime
is_production: bool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def safe_env_var(key: str, default: Optional[str] = None) -> str:
    """Safely get environment variable with robust error handling"""
    try:
        value = os.getenv(key, default)
        if value is None:
            logger.warning(f"Environment variable {key} not found, using default: {default}")
            return default or ""
        # Remove quotes and extra whitespace
        cleaned_value = value.strip().strip('"').strip("'").strip('\r').strip('\n')
        return cleaned_value
    except Exception as e:
        logger.warning(f"Error reading environment variable {key}: {e}, using default: {default}")
        return default or ""

def safe_int_env(key: str, default: int) -> int:
    """Safely parse integer environment variable with robust error handling"""
    try:
        value = os.getenv(key, str(default))
        if value is None:
            return default
        # Remove all quotes, whitespace, and newlines
        cleaned_value = value.strip().strip('"').strip("'").strip('\r').strip('\n')

        # Handle secret names that might be passed instead of actual values
        if 'latest' in cleaned_value or 'secret' in cleaned_value or ':' in cleaned_value:
            logger.warning(f"Value for {key} appears to be a secret name, using default: {default}")
            return default

        return int(cleaned_value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid value for {key}: '{value}', using default: {default}. Error: {e}")
        return default

def mask_sensitive_value(value: str, var_name: str) -> str:
    """Mask sensitive values for logging"""
    sensitive_patterns = ['API_KEY', 'SECRET', 'PASSWORD', 'TOKEN', 'ACCESS_TOKEN']
    if any(pattern in var_name.upper() for pattern in sensitive_patterns):
        if len(value) > 16:
            return f"{value[:8]}****{value[-8:]}"
        else:
            return "****"
    return value

# Main configuration loading
try:
    # Check if we're in production (Cloud Run) or development
    is_production = os.getenv('FLASK_ENV') == 'production' or os.getenv('K_SERVICE') is not None

    if is_production:
        logger.info("Running in production mode - using environment variables from Cloud Run")
        # In production, don't load .env file, use environment variables directly
    else:
        # Try to load .env file if it exists (for local development)
        env_path = project_root / '.env'
        if env_path.exists():
            logger.info(f"Loading .env file from {env_path}")
            load_dotenv(env_path, override=True)
        else:
            logger.info("No .env file found, using environment variables directly")

    # Agentic Finclusion - Phase 2 Environment Variables
    ENV_VARS: Dict[str, Union[str, int, None]] = {
        # API Keys (Required)
        'GROQ_API_KEY': None,
        'GOOGLE_API_KEY': None,

        # GCP Configuration
        'GCP_PROJECT_ID': 'linear-charmer-474120-j7',
        'GCP_PROJECT_NUMBER': '926049423219',
        'GCP_REGION': 'us-central1',

        # Database Configuration (PostgreSQL for production, SQLite for local)
        'DB_HOST': 'localhost' if not is_production else None,
        'DB_PORT': 5432,
        'DB_NAME': 'agentic_finclusion',
        'DB_USER': 'postgres',
        'DB_PASSWORD': 'oracle',
        'DB_SSLMODE': 'prefer' if not is_production else 'require',
        'DB_CONNECT_TIMEOUT': 10,

        # Flask/FastAPI Configuration
        'FLASK_ENV': 'local' if not is_production else 'production',

        # Redis Configuration (Optional - for caching)
        'REDIS_HOST': 'localhost',
        'REDIS_PORT': 6379,
        'REDIS_DB': 0,
        'REDIS_URL': 'redis://localhost:6379',
    }

    # Load environment variables with safe parsing
    config: Dict[str, Union[str, int]] = {}
    for var, default in ENV_VARS.items():
        if isinstance(default, int):
            config[var] = safe_int_env(var, default)
        else:
            config[var] = safe_env_var(var, default)

    # For local development, force Redis to localhost regardless of .env to avoid remote overrides
    # This preserves production behavior while ensuring reliable local runs
    if not is_production:
        if config.get('REDIS_HOST') != 'localhost' or not config.get('REDIS_HOST'):
            config['REDIS_HOST'] = 'localhost'
        # Always ensure local URL in development
        config['REDIS_URL'] = 'redis://localhost:6379'
        # Ensure DB index is an int and defaults to 0
        try:
            _db_val = int(str(config.get('REDIS_DB', 0)))
        except Exception:
            _db_val = 0
        config['REDIS_DB'] = _db_val

    # Log all environment variables for debugging (mask sensitive ones)
    logger.info("Environment variables loaded:")
    for var, value in config.items():
        if value:
            masked_value = mask_sensitive_value(str(value), var)
            logger.info(f"  {var}: {masked_value}")
        else:
            logger.warning(f"  {var}: NOT FOUND")

    # Check for missing critical variables
    critical_vars = ['GROQ_API_KEY', 'GOOGLE_API_KEY']
    missing_critical = [var for var in critical_vars if not config[var]]

    if missing_critical:
        logger.error(f"Missing critical environment variables: {', '.join(missing_critical)}")
        if is_production:
            logger.warning("Running in production mode with missing variables - some features may not work")
        else:
            logger.critical("Missing required environment variables")
            sys.exit(1)

    # Export all variables to global scope
    for var, value in config.items():
        globals()[var] = value

    logger.info(f"Environment: {'Production' if is_production else 'Development'}")
    logger.info("✅ Configuration loaded successfully")

except Exception as e:
    logger.critical(f"Configuration initialization failed: {str(e)}")
    # Don't exit in production, just log the error
    if not is_production:
        sys.exit(1)
