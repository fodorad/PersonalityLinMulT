---
name: Bug report
about: Report something that is broken or not working as expected
labels: bug
assignees: fodorad
---

## Description

A clear and concise description of the bug.

## Steps to reproduce

```python
# Minimal reproducible example
import torch
from personalitylinmult.models import PersonalityLinMulT

model = PersonalityLinMulT(...)
# ...
```

## Expected behaviour

What you expected to happen.

## Actual behaviour

What actually happened. Include the full traceback if applicable.

```
Traceback (most recent call last):
  ...
```

## Environment

- PersonalityLinMulT version: <!-- e.g. 3.0.0 — run `pip show personalitylinmult` -->
- PyTorch version: <!-- e.g. 2.11.0 — run `python -c "import torch; print(torch.__version__)"` -->
- Python version: <!-- e.g. 3.13.0 -->
- OS: <!-- e.g. Ubuntu 22.04 / macOS 15 / Windows 11 -->

## Additional context

Any other information that might be helpful (config dict, dataset, feature shapes, etc.).
