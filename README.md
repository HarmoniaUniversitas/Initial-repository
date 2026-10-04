# Harmonia Universitas

Research and governance framework by Gregory Elvis Graziano. This repository currently contains a foundational paper, a ψ perturbation preregistration, and exploratory Python scaffolds. It does not yet contain a completed validation of ψ or a production governance kernel.

## Start here

- [Foundational framework](HU-2026-01): guiding axiom, candidate stability functional, and verification discipline. It is a governance anchor, not an empirically verified scientific theory.
- [HU-PR-01 preregistration](%CF%88%20Coherence%20Validation%20Under%20Perturbation): historical experimental plan. Its task IDs and some execution details remain unspecified in the committed text.
- [Experiment scaffold](experiments/HU_PR_01_scaffold.py): uses a random stub `solve_task`; running it does not validate ψ on ARC tasks.
- [Coherence monitor](agents/coherence_monitor.py): exploratory implementation. Its single-observation `verify_independence` heuristic cannot establish statistical independence; its drift detector compares mean ψ across windows, not the preregistered outcome correlations.
- [Research state](docs/RESEARCH_STATE_2026-09-28.md): current evidence boundary and proposed next experiments.
- [Artifact intake](docs/ARTIFACT_INTAKE.md): requirements for bringing work from chat and external laboratories into this repository.

## Operating rule

Models reason; they do not authorize. A producer cannot validate or canonically promote its own output. Keep claims, evidence, governance decisions, and executable results distinguishable. Preserve failed and inconclusive outcomes.

## Current status

HU-PR-01 has no repository-visible real solver run or independently checked outcome bundle. Later Harmonia experiments and governance proposals discussed outside this repository are candidates for intake, not validated repository results. See the research state record before citing any result.

License: GPL-3.0 (see [LICENSE](LICENSE)).