# Research state — 2026-09-28

This is a status record, not an amendment to historical preregistrations or an approval to execute autonomous agents.

| Item | Repository evidence | State |
| --- | --- | --- |
| HU-2026-01 | Foundational paper in root | Governance anchor; candidate functional, no universal empirical validation |
| HU-PR-01 | Preregistration text and Python scaffold | Execution blocked by unspecified fixed task IDs, solver stub, and absent real results |
| Coherence monitor v1 | Python module in `agents/` | Prototype; no repository-visible independent validation |
| HU-MEK K1–K3 | Discussed in ChatGPT, source bundle not yet ingested here | Candidate intake: bounded authority, separation of duty, provenance reconstruction |
| HU-EMN-001 | Chat-reported lab skeleton and frozen preregistration | Candidate intake; no authorized autonomous agent execution evidenced here |
| HU-CADENCE / HU-SOLPI | Chat-reported sandbox and proof bundles | Candidate intake pending hashes, manifests, methods, and independent checks |
| χ / GN = 1.875 | Chat-reported parameter sweep | Engineering setpoint hypothesis in a specified model; no unique attractor or physical law established |

## Immediate corrections

1. `ReadMe` and `.github/INTEGRATION_COMPLETE.md` are historical records. Their "verified," "independence verified," and ψ = 0.72 wording should not be used as current empirical conclusions without receipts.
2. `verify_independence` in the monitor compares `e` with `1 - m` for one observation. A distance threshold is not an independence test; it can both reject independent coincident values and accept dependent values.
3. The monitor's drift test compares baseline and later mean ψ. The preregistered COD claim requires correlation of ψ with outcomes under baseline and adversarial conditions. These measures answer different questions.
4. In the scaffold, ψ uses the same run's baseline outcome as `m` and is correlated with baseline outcome. This creates target leakage for the baseline correlation. An independent predictor must be computed before, or without access to, the outcome being predicted. Any changed analysis needs a new preregistration or an explicitly labeled exploratory track.
5. The result JSON is written before `results_sha256` is inserted in memory. Thus the on-disk JSON does not contain the advertised hash. Repair prospectively, while preserving any frozen experimental specification.

## Next gated work

1. Inventory exact source bundles, version IDs, SHA-256 hashes, dates, and execution receipts for each chat-derived artifact. Record author and independent reviewer separately.
2. Choose one narrow executable slice: MEK K1 bounded authority, K2 separation of duty, K3 provenance reconstruction. Make denial and failure receipts deterministic and test bypass attempts in a capability-limited sandbox.
3. For ψ, freeze the task manifest and solver before real evaluation. Separate exploratory leakage repair from the historical HU-PR-01 claim. Publish baseline, perturbation, negative, and inconclusive outcomes with independently reproducible scripts.
4. Promote a claim only after an evidence bundle, falsification criteria, independent replication where applicable, and an explicit authorized governance decision. No automatic canonical promotion.

No cross-domain physical or mathematical claim is established by a coherent simulation alone.