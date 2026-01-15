# uvicorn src.main:app --reload --port 8000

from fastapi import FastAPI, HTTPException, Request, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, validator
from typing import Optional, Dict, Any, List
from datetime import datetime, date
import uuid
import logging
import sys
import re
from pathlib import Path
from dotenv import load_dotenv
import hashlib
import secrets
import os

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.domain.services.hybrid_classifier import HybridClassifier
from src.services.enhanced_language_detector import EnhancedLanguageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

