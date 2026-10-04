"""Model-poisoning attacks for the federated-learning laboratory."""

from .base import AttackDiagnostics
from .factory import MODEL_ATTACKS, SUPPORTED_ATTACKS, create_attack
from .selection import attack_is_active, malicious_client_ids
from .service import apply_model_attack

__all__ = [
    "AttackDiagnostics",
    "MODEL_ATTACKS",
    "SUPPORTED_ATTACKS",
    "apply_model_attack",
    "attack_is_active",
    "create_attack",
    "malicious_client_ids",
]
