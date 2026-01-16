"""
Production Monitoring Utilities for Intent Classification.

Tracks:
- Inference latency
- Confidence distributions
- Language distribution shifts
- Per-intent performance over time
- Low confidence rate (potential model degradation)
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Single inference request metrics."""
    request_id: str
    timestamp: str
    utterance_text: str
    detected_language: str
    predicted_intent: str
    confidence: float
    latency_ms: float
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    model_version: str = "v1.0.0"


@dataclass
class MonitoringWindow:
    """Aggregated metrics over a time window."""
    window_start: str
    window_end: str
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    low_confidence_rate: float = 0.0  # % below threshold
    fallback_rate: float = 0.0
    language_distribution: Dict[str, float] = field(default_factory=dict)
    intent_distribution: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProductionMonitor:
    """
    Real-time monitoring for production intent classification.
    
    Features:
    - Rolling window metrics aggregation
    - Drift detection alerts
    - Logging and persistence
    
    Production Note: In a real deployment, metrics would be sent to
    Prometheus/Grafana, DataDog, or cloud monitoring (CloudWatch, etc.)
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        log_path: str = "logs/inference_metrics.jsonl",
        alert_callback: Optional[callable] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.alert_callback = alert_callback
        
        # In-memory rolling buffer (last N requests)
        self.buffer_size = 1000
        self.metrics_buffer: List[InferenceMetrics] = []
        
        # Baseline distributions (learned from training/validation)
        self.baseline_language_dist = {"en": 0.30, "rw": 0.55, "mixed": 0.15}
        self.baseline_intent_dist: Dict[str, float] = {}
        
        # Alert thresholds
        self.alert_thresholds = {
            "low_confidence_rate": 0.20,  # Alert if >20% low confidence
            "fallback_rate": 0.15,  # Alert if >15% fallbacks
            "p95_latency_ms": 300,  # Alert if p95 latency > 300ms
            "distribution_drift": 0.15  # Alert if language dist shifts >15%
        }
    
    def record_inference(
        self,
        request_id: str,
        utterance_text: str,
        detected_language: str,
        predicted_intent: str,
        confidence: float,
        latency_ms: float,
        fallback_triggered: bool = False,
        fallback_reason: Optional[str] = None,
        model_version: str = "v1.0.0"
    ) -> InferenceMetrics:
        """Record a single inference request."""
        metrics = InferenceMetrics(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            utterance_text=utterance_text,
            detected_language=detected_language,
            predicted_intent=predicted_intent,
            confidence=confidence,
            latency_ms=latency_ms,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
            model_version=model_version
        )
        
        # Add to buffer
        self.metrics_buffer.append(metrics)
        if len(self.metrics_buffer) > self.buffer_size:
            self.metrics_buffer.pop(0)
        
        # Log to file
        self._log_metrics(metrics)
        
        # Check alerts
        self._check_alerts()
        
        return metrics
    
    def get_window_metrics(self, last_n: int = 100) -> MonitoringWindow:
        """Compute aggregated metrics over last N requests."""
        if not self.metrics_buffer:
            return MonitoringWindow(
                window_start=datetime.now().isoformat(),
                window_end=datetime.now().isoformat()
            )
        
        window = self.metrics_buffer[-last_n:]
        
        latencies = [m.latency_ms for m in window]
        confidences = [m.confidence for m in window]
        low_conf_count = sum(1 for m in window if m.confidence < self.confidence_threshold)
        fallback_count = sum(1 for m in window if m.fallback_triggered)
        
        # Language distribution
        lang_counts = defaultdict(int)
        intent_counts = defaultdict(int)
        for m in window:
            lang_counts[m.detected_language] += 1
            intent_counts[m.predicted_intent] += 1
        
        total = len(window)
        
        return MonitoringWindow(
            window_start=window[0].timestamp,
            window_end=window[-1].timestamp,
            total_requests=total,
            avg_latency_ms=statistics.mean(latencies),
            p95_latency_ms=sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0,
            avg_confidence=statistics.mean(confidences),
            low_confidence_rate=low_conf_count / total,
            fallback_rate=fallback_count / total,
            language_distribution={k: v / total for k, v in lang_counts.items()},
            intent_distribution={k: v / total for k, v in intent_counts.items()}
        )
    
    def detect_drift(self, window: MonitoringWindow) -> List[Dict[str, Any]]:
        """Detect distribution drift from baseline."""
        alerts = []
        
        # Check language distribution drift
        for lang, baseline_pct in self.baseline_language_dist.items():
            current_pct = window.language_distribution.get(lang, 0)
            drift = abs(current_pct - baseline_pct)
            
            if drift > self.alert_thresholds["distribution_drift"]:
                alerts.append({
                    "type": "language_drift",
                    "language": lang,
                    "baseline": baseline_pct,
                    "current": current_pct,
                    "drift": drift,
                    "severity": "warning" if drift < 0.25 else "critical"
                })
        
        return alerts
    
    def _check_alerts(self) -> None:
        """Check if any alert thresholds are exceeded."""
        if len(self.metrics_buffer) < 50:  # Need enough data
            return
        
        window = self.get_window_metrics(100)
        alerts = []
        
        # Low confidence rate
        if window.low_confidence_rate > self.alert_thresholds["low_confidence_rate"]:
            alerts.append({
                "type": "high_low_confidence_rate",
                "value": window.low_confidence_rate,
                "threshold": self.alert_thresholds["low_confidence_rate"],
                "severity": "warning"
            })
        
        # Fallback rate
        if window.fallback_rate > self.alert_thresholds["fallback_rate"]:
            alerts.append({
                "type": "high_fallback_rate",
                "value": window.fallback_rate,
                "threshold": self.alert_thresholds["fallback_rate"],
                "severity": "warning"
            })
        
        # Latency
        if window.p95_latency_ms > self.alert_thresholds["p95_latency_ms"]:
            alerts.append({
                "type": "high_latency",
                "value": window.p95_latency_ms,
                "threshold": self.alert_thresholds["p95_latency_ms"],
                "severity": "critical"
            })
        
        # Distribution drift
        drift_alerts = self.detect_drift(window)
        alerts.extend(drift_alerts)
        
        # Fire alerts
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")
            if self.alert_callback:
                self.alert_callback(alert)
    
    def _log_metrics(self, metrics: InferenceMetrics) -> None:
        """Append metrics to JSONL log file."""
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + "\n")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return current system health status."""
        if len(self.metrics_buffer) < 10:
            return {"status": "initializing", "requests_processed": len(self.metrics_buffer)}
        
        window = self.get_window_metrics(100)
        
        # Determine health status
        issues = []
        if window.low_confidence_rate > self.alert_thresholds["low_confidence_rate"]:
            issues.append("high_low_confidence_rate")
        if window.fallback_rate > self.alert_thresholds["fallback_rate"]:
            issues.append("high_fallback_rate")
        if window.p95_latency_ms > self.alert_thresholds["p95_latency_ms"]:
            issues.append("high_latency")
        
        if not issues:
            status = "healthy"
        elif len(issues) == 1:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "issues": issues,
            "metrics": window.to_dict()
        }


# Singleton instance for global access
_monitor_instance: Optional[ProductionMonitor] = None


def get_monitor() -> ProductionMonitor:
    """Get or create the global monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ProductionMonitor()
    return _monitor_instance
