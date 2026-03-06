# Contributing

## Setup
```bash
conda create -n cachevista python=3.11
conda activate cachevista
pip install -r requirements.txt
```

## Running tests
```bash
pytest cachevista/tests/test_core.py        # fast, no GPU needed
pytest cachevista/tests/                    # full suite, requires CLIP
```

## Linting
```bash
ruff check cachevista/
```

## Commit style
- `feat:` new feature
- `fix:` bug fix
- `test:` adding tests
- `chore:` setup, config
- `docs:` documentation
