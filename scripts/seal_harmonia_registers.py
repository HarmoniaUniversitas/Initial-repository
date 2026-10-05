import json
import hashlib
from datetime import datetime, timezone


def jcs_dumps(obj):
    return json.dumps(obj, separators=(',', ':'), sort_keys=True, ensure_ascii=False)


def seal_register(raw):
    # 1. timestamp
    raw["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw["status"] = "SEALED"

    # 2. payload_sha256 (without hash fields)
    payload_obj = {k: v for k, v in raw.items() if k not in ["payload_sha256", "self_sha256"]}
    canonical_payload = jcs_dumps(payload_obj)
    raw["payload_sha256"] = hashlib.sha256(canonical_payload.encode('utf-8')).hexdigest()

    # 3. self_sha256 (with payload hash, without self)
    self_obj = {k: v for k, v in raw.items() if k != "self_sha256"}
    canonical_self = jcs_dumps(self_obj)
    raw["self_sha256"] = hashlib.sha256(canonical_self.encode('utf-8')).hexdigest()
    return raw


# ==================== PAPER 1: HU-2026-12A ====================
hu_12A = {
  "register_id": "HU-2026-12A",
  "register_type": "PHYSICS_COSMOS_EXTENSION",
  "status": "AUTHORIZED_PENDING_SEAL",
  "version": "1.0.0",
  "canonicalization": "JCS_RFC8785",
  "timestamp_utc": None,
  "authorized_by": "Gregory E. Graziano",
  "authority_role": "Human Constitutional Authority",
  "title": "Layered Hybrid Superlattices as Room-Temperature Quantum Solids",
  "extends": "HU-2026-12",
  "anchor_ref": "HU-HSPR-2026-08-24-001:physics_cosmos_anchor",
  "claims": [
    "C12A.1: LHSLs with van der Waals molecular interlayers enable tunable charge/spin/magnetism",
    "C12A.2: Coherence time scales with interlayer crystallinity"
  ],
  "falsification": {
    "test": "Ramsey interferometry at 295K, N=20 batches",
    "FAIL_IF": "mean coherence < 1ms OR Spearman(crystallinity, coherence) < 0.3",
    "PASS_IF": "coherence >= 1ms and p<0.05"
  },
  "psi_mapping": {
    "m": "coherence_time / target_coherence",
    "e_cons": "variance across K identical stacks",
    "e_adv": "degradation at T±10K",
    "H": "entropy of stacking configurations",
    "alpha": 1.0, "beta": 1.0, "gamma": 1.0
  },
  "visual_evidence": ["Ho-Mg-Zn crystalline grids", "Fluid Tori / Vortex Imagery", "Deep-space nebulae"],
  "authority_limits": [
    "Does not claim superconductivity",
    "Room-temp claim is falsifiable, not established",
    "Does not grant canonical promotion without RE data"
  ],
  "payload_sha256": None,
  "self_sha256": None
}

# ==================== PAPER 2: HU-2026-23A ====================
hu_23A = {
  "register_id": "HU-2026-23A",
  "register_type": "GOVERNANCE_FABRICATION_EXTENSION",
  "status": "AUTHORIZED_PENDING_SEAL",
  "version": "1.0.0",
  "canonicalization": "JCS_RFC8785",
  "timestamp_utc": None,
  "authorized_by": "Gregory E. Graziano",
  "authority_role": "Human Constitutional Authority",
  "title": "Gigantic-Oxidative Atomically Layered Epitaxy as Auditable Fabrication",
  "extends": "HU-2026-23",
  "anchor_ref": "HU-HSPR-2026-08-24-001:governance_architecture_anchor",
  "claims": [
    "C23A.1: GOAL-Epitaxy enables precise complex oxide creation via enhanced oxidation",
    "C23A.2: Process is auditable via TEM defect density < 1e10 cm^-2"
  ],
  "falsification": {
    "test": "TEM + XRD across 20 runs, automated log hashing",
    "FAIL_IF": "defect_density > 1e10 cm^-2 in >30% runs",
    "PASS_IF": "scalable production without quality loss (yield variance <5%)"
  },
  "psi_mapping": {
    "m": "yield / target_yield",
    "target_yield": 0.95,
    "e_cons": "thickness variance across wafer",
    "e_adv": "stoichiometry ±2% perturbation",
    "H": "entropy of process parameters"
  },
  "audit_spine": {
    "log_hash": "SHA-256 of each epitaxy run params",
    "visual": ["Global Tree of Knowledge grid diagrams", "Regulation + and Revelation data grids"]
  },
  "authority_limits": [
    "Does not claim mass production readiness",
    "Audit requires SHA-256 log chain"
  ],
  "payload_sha256": None,
  "self_sha256": None
}

# ==================== PAPER 3: HU-2026-40A ====================
hu_40A = {
  "register_id": "HU-2026-40A",
  "register_type": "AI_DISCOVERY_BRIDGE_EXTENSION",
  "status": "AUTHORIZED_PENDING_SEAL",
  "version": "1.0.0",
  "canonicalization": "JCS_RFC8785",
  "timestamp_utc": None,
  "authorized_by": "Gregory E. Graziano",
  "authority_role": "Human Constitutional Authority",
  "title": "AI-Driven Discovery via Coherence-Gated Validation",
  "extends": ["HU-2026-40", "HU-2026-41", "HU-2026-42"],
  "anchor_ref": "HU-HSPR-2026-08-24-001:machine_ai_anchor",
  "claims": [
    "C40A.1: Deep learning identified 52k candidate layered compounds (SI)",
    "C40A.2: Only ψ > 0.7 candidates advance to synthesis (RE gating)",
    "C40A.3: Reduces experimental validation time"
  ],
  "falsification": {
    "test": "HU-PR-01 protocol: K=5 baseline, K=5 perturbed runs per candidate, N=50",
    "FAIL_IF": "Spearman(ψ, synthesis_success) < 0.3",
    "PASS_IF": "precision@psi>0.7 >= 0.6 and p<0.05"
  },
  "psi_operator": {
    "code_module": "harmonia_psi.py PsiOperator",
    "alpha": 1.0, "beta": 1.0, "gamma": 1.0,
    "w_cons": 0.5, "w_adv": 0.5,
    "threshold": 0.7,
    "storage_method": "HarmoniaMemory.store_psi_result()",
    "experiment_id": "HU-PR-01"
  },
  "visual_evidence": ["Cyborg / Humanoid Robot Imagery", "LLM Output Screenshots", "AI-generated art"],
  "authority_limits": [
    "52k compounds are SI predictions, not validated RE materials",
    "No claim of graphene-like properties until RE synthesis",
    "Does not grant AI adoption authority"
  ],
  "payload_sha256": None,
  "self_sha256": None
}


def main():
    for reg in [hu_12A, hu_23A, hu_40A]:
        sealed = seal_register(reg)
        filename = f"{sealed['register_id']}.sealed.json"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(jcs_dumps(sealed))
        print(f"SEALED {filename}")
        print(f"  payload_sha256: {sealed['payload_sha256']}")
        print(f"  self_sha256:    {sealed['self_sha256']}")
        print(f"  timestamp:      {sealed['timestamp_utc']}\n")


if __name__ == "__main__":
    main()
