"""
HU-PR-01 — ψ Coherence Validation Under Perturbation
Harmonia Universitas | Experiment ID: HU-PR-01 | Date: 2026-04-24

This scaffold matches the pre-registration document exactly.
DO NOT modify parameters after observing results.
Any change requires a new experiment ID and new pre-registration.
"""

import json
import csv
import hashlib
import numpy as np
from datetime import datetime
from scipy import stats
from dataclasses import dataclass, asdict
from typing import Optional

# ─────────────────────────────────────────────
# PRE-REGISTERED PARAMETERS — DO NOT MODIFY
# ─────────────────────────────────────────────

W1 = 0.5        # consistency error weight
W2 = 0.5        # adversarial error weight
ALPHA = 1.0     # ψ match coefficient
BETA = 1.0      # ψ error coefficient
GAMMA = 1.0     # ψ entropy coefficient
K = 5           # runs per task
N_TASKS = 25    # total tasks

# Success criteria
CORR_BASELINE_MIN = 0.3
CORR_ADV_MIN = 0.2
RATIO_MIN = 0.5

# Failure / falsification criteria
CORR_ADV_FAIL = 0.1
RATIO_FAIL = 0.5

# Fixed perturbation set (no additions permitted post-registration)
PERTURBATION_OPS = ["rotation_90", "rotation_180", "rotation_270",
                    "color_permutation", "partial_occlusion_10pct"]

# ─────────────────────────────────────────────
# CORE METRIC FUNCTIONS
# ─────────────────────────────────────────────

def compute_e_cons(outcomes_baseline: list[int]) -> float:
    """
    Consistency Error: variance of outputs across K repeated runs
    on identical input. Higher = less consistent = higher error.
    """
    return float(np.var(outcomes_baseline))


def compute_e_adv(outcome_baseline_mean: float,
                  outcome_adv_mean: float) -> float:
    """
    Adversarial Error: performance degradation between baseline
    and perturbed inputs.
    """
    return max(0.0, outcome_baseline_mean - outcome_adv_mean)


def compute_e(e_cons: float, e_adv: float) -> float:
    """
    Combined error term (pre-registered weights).
    e = w1 * e_cons + w2 * e_adv
    """
    return W1 * e_cons + W2 * e_adv


def compute_H(candidate_scores: list[float], epsilon: float = 1e-9) -> float:
    """
    Entropy of candidate solution score distribution.
    Uses normalized probability distribution over scores.
    """
    scores = np.array(candidate_scores, dtype=float)
    if scores.sum() == 0:
        return 0.0
    probs = scores / (scores.sum() + epsilon)
    probs = np.clip(probs, epsilon, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def compute_psi(m: float, e: float, H: float) -> float:
    """
    ψ = σ(α·m - β·e - γ·H)
    Pre-registered coefficients: α=β=γ=1.0
    """
    logit = ALPHA * m - BETA * e - GAMMA * H
    return float(1 / (1 + np.exp(-logit)))


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class RunRecord:
    experiment_id: str
    task_id: str
    run_index: int
    psi: float
    m: float
    e_cons: float
    e_adv: float
    e_combined: float
    H: float
    outcome_baseline: int   # {0, 1}
    outcome_adv: int        # {0, 1}
    perturbation_op: str
    timestamp: str


@dataclass
class TaskResult:
    task_id: str
    psi_mean: float
    corr_baseline: float
    corr_adv: float
    n_runs: int


# ─────────────────────────────────────────────
# PERTURBATION OPERATORS
# Implement these for your specific task format
# ─────────────────────────────────────────────

def apply_perturbation(input_grid: np.ndarray, op: str) -> np.ndarray:
    """
    Apply a fixed perturbation operator to an input grid.
    Extend per your ARC task format.
    """
    if op == "rotation_90":
        return np.rot90(input_grid, k=1)
    elif op == "rotation_180":
        return np.rot90(input_grid, k=2)
    elif op == "rotation_270":
        return np.rot90(input_grid, k=3)
    elif op == "color_permutation":
        # Fixed deterministic color swap — define your mapping here
        mapping = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7, 9: 9}
        return np.vectorize(mapping.get)(input_grid)
    elif op == "partial_occlusion_10pct":
        grid = input_grid.copy()
        n_cells = grid.size
        n_mask = max(1, int(n_cells * 0.10))
        # Deterministic masking: top-left 10%
        flat = grid.flatten()
        flat[:n_mask] = -1  # -1 = masked
        return flat.reshape(grid.shape)
    else:
        raise ValueError(f"Unknown perturbation op: {op}")


# ─────────────────────────────────────────────
# SOLVER INTERFACE
# Replace solve_task() with your actual solver
# ─────────────────────────────────────────────

def solve_task(input_grid: np.ndarray,
               ground_truth: np.ndarray,
               seed: int) -> tuple[int, list[float]]:
    """
    Stub solver. Replace with your actual ARC solver.

    Returns:
        outcome: 1 if correct, 0 if not
        candidate_scores: list of scores for entropy computation
    """
    np.random.seed(seed)
    # --- Replace below with real solver ---
    outcome = int(np.random.random() > 0.5)
    candidate_scores = list(np.random.dirichlet(np.ones(5)))
    # --- End stub ---
    return outcome, candidate_scores


# ─────────────────────────────────────────────
# MAIN EXPERIMENT LOOP
# ─────────────────────────────────────────────

def run_experiment(tasks: list[dict],
                   output_csv: str = "HU_PR_01_results.csv",
                   experiment_id: str = "HU-PR-01") -> dict:
    """
    Main experiment loop. Matches pre-registration procedure exactly.

    tasks: list of dicts with keys:
        - task_id: str
        - input_grid: np.ndarray
        - ground_truth: np.ndarray

    Returns summary dict with correlation metrics.
    """
    assert len(tasks) == N_TASKS, f"Expected {N_TASKS} tasks, got {len(tasks)}"

    all_records = []

    for task in tasks:
        task_id = task["task_id"]
        input_grid = task["input_grid"]
        ground_truth = task["ground_truth"]

        baseline_outcomes = []
        adv_outcomes = []

        # Step 1: K baseline runs
        for k in range(K):
            outcome_b, scores_b = solve_task(input_grid, ground_truth, seed=k)
            baseline_outcomes.append(outcome_b)

        # Step 2: K adversarial runs (one perturbation op per run, cycling)
        for k in range(K):
            op = PERTURBATION_OPS[k % len(PERTURBATION_OPS)]
            perturbed_grid = apply_perturbation(input_grid, op)
            outcome_a, scores_a = solve_task(perturbed_grid, ground_truth, seed=k + 1000)
            adv_outcomes.append(outcome_a)

            # Compute metrics for this run
            e_cons = compute_e_cons(baseline_outcomes[:k+1])
            e_adv = compute_e_adv(
                np.mean(baseline_outcomes[:k+1]),
                np.mean(adv_outcomes[:k+1])
            )
            e = compute_e(e_cons, e_adv)
            H = compute_H(scores_a)
            m = float(outcome_b)  # use corresponding baseline run
            psi = compute_psi(m, e, H)

            record = RunRecord(
                experiment_id=experiment_id,
                task_id=task_id,
                run_index=k,
                psi=psi,
                m=m,
                e_cons=e_cons,
                e_adv=e_adv,
                e_combined=e,
                H=H,
                outcome_baseline=baseline_outcomes[k],
                outcome_adv=adv_outcomes[k],
                perturbation_op=op,
                timestamp=datetime.utcnow().isoformat()
            )
            all_records.append(record)

    # ─────────────────────────────────────────
    # COMPUTE PRE-REGISTERED METRICS
    # ─────────────────────────────────────────

    psi_vals = np.array([r.psi for r in all_records])
    outcomes_b = np.array([r.outcome_baseline for r in all_records])
    outcomes_a = np.array([r.outcome_adv for r in all_records])

    # Use Spearman correlation (appropriate for binary outcome variable)
    corr_baseline, p_baseline = stats.spearmanr(psi_vals, outcomes_b)
    corr_adv, p_adv = stats.spearmanr(psi_vals, outcomes_a)

    ratio = corr_adv / corr_baseline if corr_baseline != 0 else 0.0

    # ─────────────────────────────────────────
    # EVALUATE AGAINST PRE-REGISTERED CRITERIA
    # ─────────────────────────────────────────

    success = (
        corr_baseline >= CORR_BASELINE_MIN and
        corr_adv >= CORR_ADV_MIN and
        ratio >= RATIO_MIN
    )

    failure = (
        corr_adv <= CORR_ADV_FAIL or
        ratio < RATIO_FAIL
    )

    summary = {
        "experiment_id": experiment_id,
        "date": datetime.utcnow().isoformat(),
        "n_tasks": N_TASKS,
        "k_runs": K,
        "n_records": len(all_records),
        "corr_baseline": round(corr_baseline, 4),
        "corr_adv": round(corr_adv, 4),
        "p_baseline": round(p_baseline, 4),
        "p_adv": round(p_adv, 4),
        "ratio": round(ratio, 4),
        "psi_valid": success,
        "psi_drifting": failure,
        "verdict": "PSI VALID" if success else ("PSI DRIFTING / INVALID" if failure else "INCONCLUSIVE")
    }

    # ─────────────────────────────────────────
    # WRITE OUTPUTS
    # ─────────────────────────────────────────

    # CSV log
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(all_records[0]).keys())
        writer.writeheader()
        for r in all_records:
            writer.writerow(asdict(r))

    # Summary JSON
    summary_path = output_csv.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Integrity hash of results
    with open(output_csv, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    summary["results_sha256"] = sha256

    print("\n" + "="*50)
    print(f"HU-PR-01 RESULTS")
    print("="*50)
    print(f"  corr(ψ, baseline):     {corr_baseline:.4f}  (p={p_baseline:.4f})")
    print(f"  corr(ψ, adversarial):  {corr_adv:.4f}  (p={p_adv:.4f})")
    print(f"  ratio (adv/base):      {ratio:.4f}")
    print(f"  VERDICT: {summary['verdict']}")
    print(f"  Results SHA256: {sha256[:16]}...")
    print("="*50)

    return summary


# ─────────────────────────────────────────────
# CALIBRATION CURVE (optional but recommended)
# ─────────────────────────────────────────────

def calibration_curve(records: list[RunRecord],
                      n_bins: int = 5) -> list[dict]:
    """
    Bucket ψ into bins, compute P(correct | ψ bin)
    for both baseline and adversarial outcomes.
    """
    psi_vals = np.array([r.psi for r in records])
    outcomes_b = np.array([r.outcome_baseline for r in records])
    outcomes_a = np.array([r.outcome_adv for r in records])

    bins = np.linspace(0, 1, n_bins + 1)
    results = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (psi_vals >= lo) & (psi_vals < hi)
        if mask.sum() == 0:
            continue
        results.append({
            "bin": f"{lo:.1f}–{hi:.1f}",
            "n": int(mask.sum()),
            "p_correct_baseline": float(outcomes_b[mask].mean()),
            "p_correct_adv": float(outcomes_a[mask].mean()),
            "psi_mean": float(psi_vals[mask].mean())
        })

    print("\nCalibration Curve:")
    print(f"{'Bin':<12} {'N':>5} {'P(correct|base)':>17} {'P(correct|adv)':>16}")
    for r in results:
        print(f"{r['bin']:<12} {r['n']:>5} {r['p_correct_baseline']:>17.3f} {r['p_correct_adv']:>16.3f}")

    return results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Replace with your actual task loader
    print("HU-PR-01 Scaffold loaded.")
    print("Replace solve_task() with your ARC solver.")
    print("Load 25 tasks into run_experiment(tasks=[...]) to execute.")
    print(f"\nPre-registered parameters:")
    print(f"  w1={W1}, w2={W2}, α={ALPHA}, β={BETA}, γ={GAMMA}, K={K}")
    print(f"  Perturbations: {PERTURBATION_OPS}")
    print(f"\nSuccess criteria:")
    print(f"  corr(ψ,base) ≥ {CORR_BASELINE_MIN}")
    print(f"  corr(ψ,adv)  ≥ {CORR_ADV_MIN}")
    print(f"  ratio        ≥ {RATIO_MIN}")
    print(f"\nFailure criteria:")
    print(f"  corr(ψ,adv)  ≤ {CORR_ADV_FAIL}  OR  ratio < {RATIO_FAIL}")
