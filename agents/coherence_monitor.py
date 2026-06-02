"""
Coherence Monitoring Agent (CMA) - v1.0
Continuous monitoring of system coherence using the ψ (psi) operator.

This agent implements the Guardian Triad architecture:
- Generator: Produces candidate outputs
- Adversary: Applies structured perturbations
- Judge: Evaluates via ψ and gates acceptance

Author: Harmonia Universitas Collaborative Agent
License: GPL-3.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import json
from datetime import datetime


class SystemStatus(Enum):
    """Coherence status classifications"""
    COHERENT = "coherent"
    DEGRADING = "degrading"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class CoherenceMetrics:
    """Container for coherence measurement across all dimensions"""
    psi: float  # Coherence amplitude
    match_score: float  # m - binary correctness
    consistency_error: float  # e_cons - variance across K runs
    adversarial_error: float  # e_adv - performance under perturbation
    entropy: float  # H - distributional entropy
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @property
    def total_error(self) -> float:
        """Combined error term (decoupled from match_score)"""
        w1, w2 = 0.5, 0.5  # Pre-registered weights
        return w1 * self.consistency_error + w2 * self.adversarial_error
    
    def to_dict(self) -> Dict:
        return {
            'psi': self.psi,
            'match_score': self.match_score,
            'consistency_error': self.consistency_error,
            'adversarial_error': self.adversarial_error,
            'entropy': self.entropy,
            'total_error': self.total_error,
            'timestamp': self.timestamp
        }


class PSIOperator:
    """
    The ψ (psi) coherence gate.
    
    Formulation:
        ψ = σ(α·m - β·e - γ·H)
    
    Where:
        m = match score (binary correctness)
        e = error term (consistency + adversarial, independent of m)
        H = entropy (distributional spread)
        σ = sigmoid activation
        α, β, γ = pre-registered coefficients
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        """
        Initialize PSI operator with pre-registered coefficients.
        
        Args:
            alpha: Weight for match score term
            beta: Weight for error term
            gamma: Weight for entropy term
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.computation_log = []
    
    @staticmethod
    def sigmoid(x: float) -> float:
        """Numerically stable sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def compute(self, metrics: CoherenceMetrics) -> float:
        """
        Compute ψ value from coherence metrics.
        
        Critical constraint: error term MUST be independent of match_score.
        Coupling breaks the dimensionality and invalidates ψ.
        """
        raw_score = (
            self.alpha * metrics.match_score
            - self.beta * metrics.total_error
            - self.gamma * metrics.entropy
        )
        psi = self.sigmoid(raw_score)
        
        # Log computation for audit trail
        self.computation_log.append({
            'timestamp': datetime.utcnow().isoformat(),
            'raw_score': raw_score,
            'psi': psi,
            'metrics': metrics.to_dict()
        })
        
        return psi
    
    def verify_independence(self, metrics: CoherenceMetrics) -> Tuple[bool, str]:
        """
        Verify that error term is independent of match_score.
        Returns (is_valid, reason).
        
        This is the mathematical guard against ψ drift.
        """
        if metrics.match_score < 0 or metrics.match_score > 1:
            return False, f"match_score out of bounds: {metrics.match_score}"
        
        if metrics.total_error < 0 or metrics.total_error > 1:
            return False, f"total_error out of bounds: {metrics.total_error}"
        
        # Flag coupling: if e ≈ 1 - m, the operator collapses to 1D
        coupling_ratio = abs(metrics.total_error - (1.0 - metrics.match_score))
        if coupling_ratio < 0.1:
            return False, f"Error-Match coupling detected (ratio: {coupling_ratio})"
        
        return True, "Independence verified"


class CoherenceTrendAnalyzer:
    """Analyzes temporal trends in coherence to detect drift"""
    
    def __init__(self, window_size: int = 10, drift_threshold: float = 0.2):
        """
        Args:
            window_size: Rolling window for trend analysis
            drift_threshold: Tolerance for acceptable degradation
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.history: List[CoherenceMetrics] = []
    
    def add_measurement(self, metrics: CoherenceMetrics) -> None:
        """Record a new coherence measurement"""
        self.history.append(metrics)
    
    def detect_psi_drift(self) -> Tuple[bool, float, str]:
        """
        Detect Coherence-Outcome Divergence (COD):
        whether ψ correlation with outcomes degrades under adversarial conditions.
        
        Returns:
            (is_drifting, degradation_rate, status_message)
        """
        if len(self.history) < 2:
            return False, 0.0, "Insufficient history for drift detection"
        
        # Calculate baseline correlation window
        baseline_window = self.history[:self.window_size]
        baseline_mean = np.mean([m.psi for m in baseline_window])
        baseline_std = np.std([m.psi for m in baseline_window])
        
        if baseline_std == 0:
            return False, 0.0, "Baseline has zero variance"
        
        # Calculate adversarial window (recent measurements)
        if len(self.history) <= self.window_size:
            return False, 0.0, "Waiting for adversarial window"
        
        adv_window = self.history[self.window_size:]
        adv_mean = np.mean([m.psi for m in adv_window])
        
        # Degradation rate: normalized drop in mean ψ
        degradation = (baseline_mean - adv_mean) / baseline_mean if baseline_mean > 0 else 0
        
        is_drifting = degradation > self.drift_threshold
        
        status = f"Baseline ψ: {baseline_mean:.3f}, Adversarial ψ: {adv_mean:.3f}, Degradation: {degradation:.3f}"
        
        return is_drifting, degradation, status
    
    def get_status(self) -> SystemStatus:
        """Classify current system status based on recent coherence trends"""
        if len(self.history) == 0:
            return SystemStatus.UNKNOWN
        
        recent_psi = self.history[-1].psi
        
        if recent_psi > 0.7:
            return SystemStatus.COHERENT
        elif recent_psi > 0.4:
            return SystemStatus.DEGRADING
        else:
            return SystemStatus.CRITICAL


class GuardianTriad:
    """
    Three-role architecture for coherence-gated system evolution:
    - G₁ (Generator): Produces candidate outputs
    - G₂ (Adversary): Applies structured perturbations
    - G₃ (Judge): Evaluates via ψ
    """
    
    def __init__(self, 
                 generator: Callable,
                 adversary: Callable,
                 evaluator: Callable):
        """
        Args:
            generator: Function(psi_state) -> candidate_output
            adversary: Function(output) -> perturbed_output
            evaluator: Function(output, expected) -> CoherenceMetrics
        """
        self.G1 = generator
        self.G2 = adversary
        self.G3 = evaluator
        self.psi_op = PSIOperator()
        self.iteration_log = []
    
    def evaluate_cycle(self, expected_output, psi_state: float) -> Tuple[float, CoherenceMetrics]:
        """
        Single cycle of the Guardian Triad:
        1. G₁ generates candidate
        2. G₂ applies perturbation
        3. G₃ evaluates both baseline and perturbed versions
        4. Judge gates decision via ψ
        
        Returns:
            (new_psi, metrics)
        """
        # G₁: Generate candidate
        candidate = self.G1(psi_state)
        
        # G₃: Evaluate baseline
        baseline_metrics = self.G3(candidate, expected_output)
        
        # G₂: Apply adversarial perturbation
        perturbed = self.G2(candidate)
        
        # G₃: Evaluate under perturbation
        adv_metrics = self.G3(perturbed, expected_output)
        
        # Blend metrics: weighted combination of baseline and adversarial
        combined_metrics = CoherenceMetrics(
            psi=0.0,  # Will be computed
            match_score=0.8 * baseline_metrics.match_score + 0.2 * adv_metrics.match_score,
            consistency_error=0.5 * baseline_metrics.consistency_error + 0.5 * adv_metrics.consistency_error,
            adversarial_error=adv_metrics.adversarial_error,
            entropy=baseline_metrics.entropy
        )
        
        # Judge: Compute ψ
        new_psi = self.psi_op.compute(combined_metrics)
        combined_metrics.psi = new_psi
        
        # Log
        self.iteration_log.append({
            'baseline': baseline_metrics.to_dict(),
            'adversarial': adv_metrics.to_dict(),
            'combined_psi': new_psi,
            'gating_decision': 'accept' if new_psi > 0.5 else 'recurse'
        })
        
        return new_psi, combined_metrics


class CoherenceMonitoringAgent:
    """
    Main coherence monitoring agent orchestrating continuous surveillance.
    Implements both real-time monitoring and batch analysis modes.
    """
    
    def __init__(self, 
                 guardian_triad: GuardianTriad,
                 alert_threshold: float = 0.3,
                 window_size: int = 10):
        """
        Args:
            guardian_triad: Guardian Triad evaluator
            alert_threshold: PSI threshold below which alerts trigger
            window_size: Measurement history for trend analysis
        """
        self.triad = guardian_triad
        self.alert_threshold = alert_threshold
        self.analyzer = CoherenceTrendAnalyzer(window_size=window_size)
        self.alerts = []
        self.current_psi = 0.5  # Start at neutral coherence
        self.measurement_count = 0
    
    def monitor_cycle(self, expected_output, perturbation_set: Optional[List] = None) -> Dict:
        """
        Execute one monitoring cycle:
        1. Run Guardian Triad evaluation
        2. Update coherence trends
        3. Check for drift
        4. Generate alerts if needed
        
        Returns:
            cycle_report with metrics, status, alerts
        """
        # Generate new PSI
        new_psi, metrics = self.triad.evaluate_cycle(expected_output, self.current_psi)
        self.current_psi = new_psi
        self.measurement_count += 1
        
        # Add to trend history
        self.analyzer.add_measurement(metrics)
        
        # Detect drift
        is_drifting, degradation, drift_status = self.analyzer.detect_psi_drift()
        
        # Determine status
        status = self.analyzer.get_status()
        
        # Generate alerts
        cycle_alerts = []
        if new_psi < self.alert_threshold:
            alert = {
                'type': 'LOW_COHERENCE',
                'severity': 'CRITICAL' if new_psi < 0.2 else 'WARNING',
                'psi': new_psi,
                'timestamp': datetime.utcnow().isoformat(),
                'message': f'PSI below threshold: {new_psi:.3f} < {self.alert_threshold}'
            }
            cycle_alerts.append(alert)
            self.alerts.append(alert)
        
        if is_drifting:
            alert = {
                'type': 'PSI_DRIFT',
                'severity': 'CRITICAL',
                'degradation': degradation,
                'timestamp': datetime.utcnow().isoformat(),
                'message': drift_status
            }
            cycle_alerts.append(alert)
            self.alerts.append(alert)
        
        # Verify independence
        is_valid, reason = self.triad.psi_op.verify_independence(metrics)
        if not is_valid:
            alert = {
                'type': 'INDEPENDENCE_VIOLATION',
                'severity': 'CRITICAL',
                'timestamp': datetime.utcnow().isoformat(),
                'reason': reason
            }
            cycle_alerts.append(alert)
            self.alerts.append(alert)
        
        return {
            'cycle': self.measurement_count,
            'psi': new_psi,
            'metrics': metrics.to_dict(),
            'status': status.value,
            'is_drifting': is_drifting,
            'degradation': degradation,
            'alerts': cycle_alerts,
            'independence_valid': is_valid
        }
    
    def get_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        if len(self.analyzer.history) == 0:
            return {'status': 'NO_DATA', 'measurements': 0}
        
        psi_values = [m.psi for m in self.analyzer.history]
        
        return {
            'total_measurements': self.measurement_count,
            'current_psi': self.current_psi,
            'mean_psi': np.mean(psi_values),
            'std_psi': np.std(psi_values),
            'min_psi': np.min(psi_values),
            'max_psi': np.max(psi_values),
            'system_status': self.analyzer.get_status().value,
            'total_alerts': len(self.alerts),
            'critical_alerts': len([a for a in self.alerts if a['severity'] == 'CRITICAL']),
            'alert_history': self.alerts
        }
