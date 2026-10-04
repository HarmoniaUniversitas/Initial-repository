# Chat-to-repository artifact intake

Each proposed import must have an ID and status (`DRAFT`, `CANDIDATE`, `VALIDATED`, `REPLICATED`, or `CANONICAL`). Status is assigned by an authorized reviewer; a model or producer cannot self-validate. Governance anchors and scientific claims must have separate claim types.

Required manifest fields: artifact ID and version; title and claim type; source and author; date; immutable source bytes and SHA-256; dependency IDs; methods and environment; exact commands and seed/config; raw outputs and checksums; known limitations; falsification condition; producer; independent reviewer; authority decision and date. Missing fields remain visibly `UNKNOWN` and block promotion.

The intake sequence is: preserve source → reconstruct provenance → classify claims → run schema and capability checks → reproduce in isolation → independent review → explicit decision. A failed check generates a receipt and leaves the item at its previous state. An amended item receives a new version and retains links to the old one.

Do not commit secrets, personal data, or unreviewed chat transcripts. Summaries of chat work must say "chat-reported" until the underlying files and receipts are present and checked. In particular, a frozen preregistration permits no quiet parameter retuning, and a lab skeleton does not imply authorization for agent execution.

## Follow-up implementation pull request

Once the actual HU-MEK and recent sandbox bundles are retrieved, add their original manifests and hashes, then implement only the minimum kernel and adversarial tests. Keep the ψ experiment repair on a separate branch so historical preregistration and exploratory revision are never conflated.