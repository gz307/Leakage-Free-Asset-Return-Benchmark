# Third-party model source attribution

This reproduction package contains only the upstream source files imported by the study's model adapters. The applicable upstream license files are retained; full upstream documentation remains available from the original repositories linked below.

- Autoformer: https://github.com/thuml/Autoformer
- PatchTST: https://github.com/yuqinie98/PatchTST
- iTransformer: https://github.com/thuml/iTransformer

The study adapts only the input/output interfaces required for univariate one-step-ahead forecasting; the main architectural components are retained as described in the manuscript.

Compatibility note: `iTransformer/layers/SelfAttention_Family.py` originally imports `reformer_pytorch` at module import time. The study uses `FullAttention`, not `LSHSelfAttention`; therefore the supplied snapshot makes the unused Reformer dependency optional so the reported iTransformer configuration can be imported without installing an otherwise unused package. No reported model computation is changed by this import-compatibility patch.
