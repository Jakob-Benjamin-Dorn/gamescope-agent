# Migration from the educational notebook

The public repository intentionally excludes:

- the original course instructions and rubric text;
- the course-provided `lib/` framework;
- Vocareum-specific API endpoints;
- executed notebook outputs containing old workspace paths;
- hard-coded assumptions that the collection must contain exactly 15 items.

The portfolio version adds:

- a `src/` package layout;
- environment-based configuration;
- idempotent ingestion;
- explicit evidence-first orchestration;
- session memory and workflow traces;
- deterministic unit tests without paid API calls;
- CI, Ruff configuration, licensing, and attribution notes.
