# Research Method & Falsification

PTR is simultaneously software and architecture research. Claims must survive:

1. precise hypothesis;
2. matched baseline;
3. controlled ablation;
4. multiple seeds/tasks;
5. OOD and long-horizon evaluation;
6. failure/negative result recording;
7. prior-art update;
8. reproducible artifacts.

Examples of hypotheses:
- semantic slots reduce hard-constraint violations at equal compute;
- latent recurrence reduces output tokens without reducing task success;
- operator routing improves specialist tasks vs language-only reasoning;
- semantic project state beats strong RAG on editable long-horizon knowledge;
- generation-aware authority prevents stale resurrection under crash/restart.

A failed hypothesis is a valid research result and should remove or simplify architecture.
