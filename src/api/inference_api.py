"""
Production Inference API for Irembo Voice AI Intent Classification.

Features:
- Transformer-based primary classifier
- Confidence-based fallback to rule-based system
- Groq LLM as third-tier fallback for edge cases
- Request logging and monitoring
- Health endpoints for orchestration
"""

import os
import sys
import uuid
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env from project root (handles various working directories)
_project_root = Path(__file__).resolve().parents[2]
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)
else:
    # Fallback: try current working directory
    load_dotenv(override=True)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from ..domain.services.rule_based_classifier import RuleBasedClassifier
from ..domain.services.groq_classifier import GroqClassifier
from ..services.enhanced_language_detector import EnhancedLanguageDetector
from ..config.monitoring import get_monitor, ProductionMonitor
from ..config.model_registry import ModelRegistry, DEPLOYMENT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Global Model Instances (loaded at startup)
# ============================================================================

class ModelContainer:
    """Container for loaded models."""
    transformer_model = None
    tokenizer = None
    label_map: Dict[int, str] = {}
    rule_classifier: Optional[RuleBasedClassifier] = None
    groq_classifier: Optional[GroqClassifier] = None
    groq_enabled: bool = False
    language_detector: Optional[EnhancedLanguageDetector] = None
    monitor: Optional[ProductionMonitor] = None
    model_version: str = "v1.0.0"
    device: str = "cpu"

models = ModelContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup at shutdown."""
    logger.info("Loading models...")
    
    # Determine device
    models.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {models.device}")
    
    # Load transformer model
    model_path = Path(DEPLOYMENT_CONFIG["model_path"])
    if model_path.exists():
        models.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        models.transformer_model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        ).to(models.device)
        models.transformer_model.eval()
        
        # Build label map from model config
        if hasattr(models.transformer_model.config, 'id2label'):
            models.label_map = models.transformer_model.config.id2label
        else:
            # Fallback: load from training config
            import json
            config_file = model_path / "training_config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                    if "label2id" in config:
                        models.label_map = {v: k for k, v in config["label2id"].items()}
        
        logger.info(f"Loaded transformer model with {len(models.label_map)} labels")
    else:
        logger.warning(f"Model not found at {model_path}, using fallback only")
    
    # Load fallback classifiers
    models.rule_classifier = RuleBasedClassifier()
    models.language_detector = EnhancedLanguageDetector()
    models.monitor = get_monitor()
    
    # Load Groq LLM classifier (third-tier fallback)
    try:
        models.groq_classifier = GroqClassifier()
        models.groq_enabled = True
        logger.info("Groq LLM classifier loaded (third-tier fallback)")
    except Exception as e:
        logger.warning(f"Groq classifier not available: {e}")
        models.groq_enabled = False
    
    # Get model version from registry
    registry = ModelRegistry()
    active = registry.get_active_model("intent_classifier")
    if active:
        models.model_version = active.version
    
    logger.info("Models loaded successfully")
    
    yield  # Application runs here
    
    # Cleanup
    logger.info("Shutting down, cleaning up resources...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Irembo Voice AI Intent Classifier",
    description="Production API for multilingual intent classification (Kinyarwanda/English)",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class ClassifyRequest(BaseModel):
    """Intent classification request."""
    utterance_text: str = Field(..., description="User utterance to classify", min_length=1)
    language_hint: Optional[str] = Field(None, description="Optional language hint (en, rw, mixed)")

class ClassifyResponse(BaseModel):
    """Intent classification response."""
    request_id: str
    utterance_text: str
    detected_language: str
    predicted_intent: str
    confidence: float
    fallback_used: bool
    fallback_reason: Optional[str] = None
    model_version: str
    latency_ms: float

class BatchClassifyRequest(BaseModel):
    """Batch classification request."""
    utterances: List[str] = Field(..., min_items=1, max_items=100)

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    groq_enabled: bool = False
    model_version: str
    device: str
    metrics: Optional[Dict[str, Any]] = None


# ============================================================================
# Inference Logic
# ============================================================================

def classify_with_transformer(text: str) -> tuple[str, float]:
    """Run transformer inference, return (intent, confidence)."""
    if models.transformer_model is None or models.tokenizer is None:
        raise ValueError("Transformer model not loaded")
    
    inputs = models.tokenizer(
        text,
        truncation=True,
        max_length=DEPLOYMENT_CONFIG["max_sequence_length"],
        return_tensors="pt"
    ).to(models.device)
    
    with torch.no_grad():
        outputs = models.transformer_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, pred_idx = torch.max(probs, dim=-1)
    
    intent = models.label_map.get(pred_idx.item(), "unknown")
    return intent, confidence.item()


def classify_with_groq(text: str, language: str = "en") -> tuple[str, float]:
    """Run Groq LLM inference, return (intent, confidence)."""
    if models.groq_classifier is None:
        raise ValueError("Groq classifier not loaded")
    
    result = models.groq_classifier.classify(text, language)
    return result.intent.value, result.confidence


def classify_with_fallback(text: str, language: str = "en") -> tuple[str, float, bool, Optional[str]]:
    """
    Classify with three-tier confidence-based fallback strategy.
    
    Tier 1: Transformer (primary) - fast, accurate, self-contained
    Tier 2: Rule-based (secondary) - interpretable, no external deps
    Tier 3: Groq LLM (tertiary) - handles edge cases, external API
    
    Returns:
        (intent, confidence, fallback_used, fallback_reason)
    """
    fallback_used = False
    fallback_reason = None
    confidence_threshold = DEPLOYMENT_CONFIG["confidence_threshold"]
    
    best_intent = None
    best_confidence = 0.0
    
    # ========== TIER 1: Transformer ==========
    if models.transformer_model is not None:
        try:
            intent, confidence = classify_with_transformer(text)
            best_intent, best_confidence = intent, confidence
            
            # High confidence - return immediately
            if confidence >= confidence_threshold:
                return intent, confidence, False, None
                
        except Exception as e:
            logger.error(f"Transformer inference failed: {e}")
            fallback_reason = f"transformer_error: {str(e)}"
    else:
        fallback_reason = "transformer_not_loaded"
    
    # ========== TIER 2: Rule-based ==========
    fallback_used = True
    rule_result = models.rule_classifier.classify(text, language)
    
    if rule_result.confidence > best_confidence:
        best_intent = rule_result.intent.value
        best_confidence = rule_result.confidence
        fallback_reason = f"rule_based_confidence_{rule_result.confidence:.3f}"
    
    # If rule-based has high confidence, return
    if best_confidence >= confidence_threshold:
        return best_intent, best_confidence, True, fallback_reason
    
    # ========== TIER 3: Groq LLM ==========
    if models.groq_enabled and models.groq_classifier is not None:
        try:
            groq_intent, groq_confidence = classify_with_groq(text, language)
            
            if groq_confidence > best_confidence:
                best_intent = groq_intent
                best_confidence = groq_confidence
                fallback_reason = f"groq_llm_confidence_{groq_confidence:.3f}"
                logger.info(f"Groq LLM used for: '{text[:50]}...' -> {groq_intent} ({groq_confidence:.3f})")
                
        except Exception as e:
            logger.warning(f"Groq LLM fallback failed: {e}")
            # Keep best result from earlier tiers
    
    # Return best result from all tiers
    if best_intent is None:
        best_intent = "unknown"
        best_confidence = 0.0
        fallback_reason = "all_classifiers_failed"
    
    return best_intent, best_confidence, fallback_used, fallback_reason


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def demo_ui():
    """Serve the demo HTML UI."""
    demo_path = Path(__file__).with_name("demo.html")
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="Demo UI not found")
    return HTMLResponse(demo_path.read_text(encoding="utf-8"))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for orchestration/load balancers."""
    health = models.monitor.get_health_status() if models.monitor else {"status": "unknown"}
    
    return HealthResponse(
        status=health.get("status", "unknown"),
        model_loaded=models.transformer_model is not None,
        groq_enabled=models.groq_enabled,
        model_version=models.model_version,
        device=models.device,
        metrics=health.get("metrics")
    )


@app.get("/ready")
async def readiness_check():
    """Kubernetes-style readiness probe."""
    if models.transformer_model is None and models.rule_classifier is None:
        raise HTTPException(status_code=503, detail="No models loaded")
    return {"ready": True}


@app.post("/classify", response_model=ClassifyResponse)
async def classify_intent(request: ClassifyRequest):
    """
    Classify user intent from utterance text.
    
    Three-tier fallback strategy:
    1. Transformer (primary) - high accuracy, self-contained
    2. Rule-based (secondary) - fast, interpretable
    3. Groq LLM (tertiary) - handles edge cases
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Detect language
    language = request.language_hint or "unknown"
    if language == "unknown" and models.language_detector:
        lang_result = models.language_detector.detect(request.utterance_text)
        language = lang_result.get("language", "unknown")
    
    # Classify with three-tier fallback logic
    intent, confidence, fallback_used, fallback_reason = classify_with_fallback(
        request.utterance_text, language
    )
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Record metrics
    if models.monitor:
        models.monitor.record_inference(
            request_id=request_id,
            utterance_text=request.utterance_text,
            detected_language=language,
            predicted_intent=intent,
            confidence=confidence,
            latency_ms=latency_ms,
            fallback_triggered=fallback_used,
            fallback_reason=fallback_reason,
            model_version=models.model_version
        )
    
    return ClassifyResponse(
        request_id=request_id,
        utterance_text=request.utterance_text,
        detected_language=language,
        predicted_intent=intent,
        confidence=round(confidence, 4),
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        model_version=models.model_version,
        latency_ms=round(latency_ms, 2)
    )


@app.post("/classify/batch")
async def classify_batch(request: BatchClassifyRequest):
    """
    Batch classification endpoint for offline/bulk processing.
    """
    results = []
    for text in request.utterances:
        req = ClassifyRequest(utterance_text=text)
        result = await classify_intent(req)
        results.append(result.model_dump())
    
    return {
        "count": len(results),
        "results": results
    }


@app.get("/model/info")
async def model_info():
    """Return model metadata and configuration."""
    registry = ModelRegistry()
    active = registry.get_active_model("intent_classifier")
    versions = registry.list_versions("intent_classifier")
    
    return {
        "active_version": active.to_dict() if active else None,
        "all_versions": versions,
        "deployment_config": {
            k: v for k, v in DEPLOYMENT_CONFIG.items()
            if k != "environment"  # Don't expose internal configs
        },
        "labels": models.label_map
    }


@app.get("/metrics")
async def get_metrics():
    """Return current monitoring metrics."""
    if models.monitor is None:
        raise HTTPException(status_code=503, detail="Monitoring not initialized")
    
    window = models.monitor.get_window_metrics(100)
    return window.to_dict()


# ============================================================================
# Run with: uvicorn src.api.inference_api:app --reload --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
