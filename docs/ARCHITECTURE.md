# SceneSift — Proposed Architecture

> These diagrams describe a prospective design, not an existing implementation.

## End-to-end processing

```mermaid
flowchart TD
    I[Input video] --> D[Decode and align original timestamps]
    P[Optional natural-language request] --> Q[Text embedding / query representation]
    D --> C[Scene changes plus overlapping windows]
    C --> S[Candidate segments]
    S --> V[Sample representative frames]
    S --> M[Motion measurements]
    S --> A[Audio windows]
    V --> VE[Visual embeddings]
    A --> AE[Audio event features]
    A --> ST[Speech transcription if available]
    ST --> TE[Transcript relevance and topic novelty]
    VE --> N[Timestamp alignment and normalization]
    M --> N
    AE --> N
    TE --> N
    Q --> N
    N --> R[Multimodal highlight scoring]
    R --> H[Select relevant, diverse segments under duration budget]
    H --> X[Add context and assemble timeline]
    X --> O[Highlight MP4]
    H --> T[Source timestamps and scores]
    O --> UI[Preview and download interface]
    T --> UI
```

## Research comparison

```mermaid
flowchart LR
    D[Same held-out annotated videos] --> A[Baseline A: activity signals]
    D --> B[Baseline B: visual semantics]
    D --> C[Proposed: multimodal ranking]
    A --> E[Equal summary-duration budgets]
    B --> E
    C --> E
    E --> F[Temporal F1, event coverage, redundancy, human relevance, runtime]
    F --> G[Per-category results and failure analysis]
```

## Architecture principles

- **Category-independent:** The same candidate/feature/scoring structure is applied to basketball, gaming, podcasts, and travel videos. Category labels are for evaluation, not hardcoded detection rules.
- **Temporal coherence:** Scene boundaries alone are insufficient; overlapping windows preserve events occurring within long shots.
- **Missing evidence:** Silent videos or missing speech bypass unavailable branches. Remaining feature contributions are renormalized.
- **Prompt guidance:** A request biases relevance but is not guaranteed to identify every complex or multi-frame event.
- **Non-redundancy:** Selected clips must jointly cover distinct useful moments within the configured duration budget.
- **Fair evaluation:** Compared methods use the same test videos and budgets, with no tuning on final test annotations.

See [README](../README.md), [Experiment](EXPERIMENT.md), and [Implementation](IMPLEMENTATION.md).
