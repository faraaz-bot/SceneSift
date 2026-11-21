# SceneSift — Architecture

> These diagrams describe the implemented pipeline. See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the module behind each box. In the research comparison, the classical method, the blend, and the TVSum run are not implemented yet.

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
    D[SumMe and TVSum videos with human labels] --> U[Uniform reference]
    D --> A[Baseline A: activity signals]
    D --> B[Baseline B: visual semantics]
    D --> C[Proposed: multimodal ranking]
    D --> K[Planned D: classical CV only]
    D --> L[Planned: blend, weights fitted on SumMe]
    U --> E[Equal summary-duration budgets]
    A --> E
    B --> E
    C --> E
    K --> E
    L --> E
    E --> F[Temporal F1, event coverage, redundancy, rank correlation, runtime]
    F --> G[Per-category TVSum results and failure analysis]
```

## Architecture principles

- **Category-independent:** The same candidate/feature/scoring structure is applied to every video, whatever its category. Category labels are for evaluation, not hardcoded detection rules.
- **Temporal coherence:** Scene boundaries alone are insufficient; overlapping windows preserve events occurring within long shots.
- **Missing evidence:** Silent videos or missing speech bypass unavailable branches. Remaining feature contributions are renormalized.
- **Prompt guidance:** A request biases relevance but is not guaranteed to identify every complex or multi-frame event.
- **Non-redundancy:** Selected clips must jointly cover distinct useful moments within the configured duration budget.
- **Fair evaluation:** Compared methods use the same test videos and budgets, with weights fitted on SumMe only and never on TVSum, the test set.

See [README](../README.md), [Experiment](EXPERIMENT.md), and [Implementation](IMPLEMENTATION.md).
