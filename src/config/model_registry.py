"""
Model Registry and Versioning Configuration.

Provides structured tracking for model artifacts, versions, and metadata.
Enables reproducibility and rollback in production deployments.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List


@dataclass
class ModelVersion:
    """Represents a versioned model artifact."""
    version: str
    model_name: str
    model_path: str
    created_at: str
    training_config: Dict[str, Any]
    metrics: Dict[str, float]
    data_hash: str  # Hash of training data for reproducibility
    description: str = ""
    is_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        return cls(**data)


class ModelRegistry:
    """
    Simple file-based model registry for versioning and deployment tracking.
    
    Production Note: In a real deployment, this would integrate with
    MLflow, DVC, or a cloud model registry (SageMaker, Vertex AI).
    """
    
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load existing registry or initialize empty."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                self.models = {
                    k: [ModelVersion.from_dict(v) for v in versions]
                    for k, versions in data.get("models", {}).items()
                }
                self.active_versions = data.get("active_versions", {})
        else:
            self.models: Dict[str, List[ModelVersion]] = {}
            self.active_versions: Dict[str, str] = {}
    
    def _save_registry(self) -> None:
        """Persist registry to disk."""
        data = {
            "models": {
                k: [v.to_dict() for v in versions]
                for k, versions in self.models.items()
            },
            "active_versions": self.active_versions,
            "last_updated": datetime.now().isoformat()
        }
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        training_config: Dict[str, Any],
        metrics: Dict[str, float],
        data_path: str,
        description: str = "",
        set_active: bool = True
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_name: Logical name (e.g., "intent_classifier")
            model_path: Path to model artifacts
            training_config: Hyperparameters and training settings
            metrics: Evaluation metrics (accuracy, f1, etc.)
            data_path: Path to training data (for hashing)
            description: Human-readable description
            set_active: Whether to make this the active version
        
        Returns:
            The created ModelVersion
        """
        # Generate version string
        existing = self.models.get(model_name, [])
        version_num = len(existing) + 1
        version = f"v{version_num}.0.0"
        
        # Hash training data for reproducibility
        data_hash = self._compute_data_hash(data_path)
        
        model_version = ModelVersion(
            version=version,
            model_name=model_name,
            model_path=model_path,
            created_at=datetime.now().isoformat(),
            training_config=training_config,
            metrics=metrics,
            data_hash=data_hash,
            description=description,
            is_active=set_active
        )
        
        if model_name not in self.models:
            self.models[model_name] = []
        
        # Deactivate previous versions if setting this as active
        if set_active:
            for v in self.models[model_name]:
                v.is_active = False
            self.active_versions[model_name] = version
        
        self.models[model_name].append(model_version)
        self._save_registry()
        
        return model_version
    
    def get_active_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get the currently active model version."""
        if model_name not in self.models:
            return None
        
        for v in self.models[model_name]:
            if v.is_active:
                return v
        return None
    
    def get_model_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        if model_name not in self.models:
            return None
        
        for v in self.models[model_name]:
            if v.version == version:
                return v
        return None
    
    def rollback(self, model_name: str, version: str) -> bool:
        """Rollback to a previous model version."""
        target = self.get_model_version(model_name, version)
        if not target:
            return False
        
        for v in self.models[model_name]:
            v.is_active = (v.version == version)
        
        self.active_versions[model_name] = version
        self._save_registry()
        return True
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model."""
        if model_name not in self.models:
            return []
        
        return [
            {
                "version": v.version,
                "created_at": v.created_at,
                "metrics": v.metrics,
                "is_active": v.is_active,
                "description": v.description
            }
            for v in self.models[model_name]
        ]
    
    def _compute_data_hash(self, data_path: str) -> str:
        """Compute hash of training data for reproducibility."""
        path = Path(data_path)
        if not path.exists():
            return "unknown"
        
        hasher = hashlib.sha256()
        if path.is_file():
            with open(path, 'rb') as f:
                hasher.update(f.read())
        else:
            # Hash directory contents
            for file in sorted(path.glob("**/*")):
                if file.is_file():
                    hasher.update(file.name.encode())
        
        return hasher.hexdigest()[:16]


# Deployment configuration
DEPLOYMENT_CONFIG = {
    "model_name": "intent_classifier",
    "model_type": "transformer",
    "base_model": "Davlan/afro-xlmr-base",
    "model_path": "models/transformer_intent",
    "max_sequence_length": 128,
    "batch_size": 32,
    "inference_timeout_ms": 500,
    "confidence_threshold": 0.7,  # Below this, trigger fallback
    "fallback_strategy": "rule_based",  # Options: rule_based, human_escalation
    "supported_languages": ["en", "rw", "mixed"],
    "version": "v1.0.0",
    "environment": {
        "development": {
            "replicas": 1,
            "gpu_enabled": False,
            "logging_level": "DEBUG"
        },
        "staging": {
            "replicas": 2,
            "gpu_enabled": False,
            "logging_level": "INFO"
        },
        "production": {
            "replicas": 4,
            "gpu_enabled": True,
            "logging_level": "WARNING"
        }
    }
}


if __name__ == "__main__":
    # Example: Register the trained model
    registry = ModelRegistry()
    
    # Load training config
    config_path = Path("models/transformer_intent/training_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            training_config = json.load(f)
    else:
        training_config = {"base_model": "Davlan/afro-xlmr-base"}
    
    # Register with actual metrics
    version = registry.register_model(
        model_name="intent_classifier",
        model_path="models/transformer_intent",
        training_config=training_config,
        metrics={
            "accuracy": 0.9565,
            "macro_f1": 0.9563,
            "macro_precision": 0.9654,
            "macro_recall": 0.9526
        },
        data_path="data/voiceai_intent_train.csv",
        description="Fine-tuned AfroXLMR on Irembo intent data with 13 intents"
    )
    
    print(f"Registered model: {version.model_name} {version.version}")
    print(f"Metrics: {version.metrics}")
