# Repository Guidelines

## Project Structure & Module Organization
This repository curates PLC Structured Text (ST) corpora and utilities used to build retrieval-ready metadata. Core processing scripts live at the repository root (`split_json.py`, `process_chunks.py`, `merge_metadata.py`, `validate_merge.py`). Supporting tokenizer assets and helper code live in `code/`, including the DeepSeek tokenizer bundle under `code/deepseek_v3_tokenizer/`. Training data and derived artifacts sit in `data/` (notably `data/st_train.json`, chunk shards in `data/chunks/`, and merged outputs in `data/chunks_metadata_*.json`). Reference manuals used for grounding live in `books/`. Treat `empty.py` as a placeholder for new modules.

## Build, Test, and Development Commands
- `python split_json.py`: preprocesses `data/st_train.json` into chunked JSON files under `data/chunks/`.
- `python process_chunks.py`: enriches each chunk with ST metadata, producing `data/chunks_metadata_all_in_one_XXX.json`.
- `python merge_metadata.py`: combines the generated metadata shards into `data/chunks_metadata_complete_all.json`.
- `python validate_merge.py`: inspects the merged file and reports field coverage.
Run commands from the repository root with Python >=3.10; prefer virtual environments (`python -m venv .venv && source .venv/bin/activate`) to manage dependencies.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, type hints for public helpers, and `snake_case` for functions, module names, and JSON keys. Keep docstrings short, bilingual where practical, and focus comments on PLC/ST domain intent. When adjusting hard-coded paths, prefer `pathlib.Path` and repository-relative joins to ease portability. Before submitting, format files with `black` (or `ruff format`) and lint with `ruff` if available.

## Testing Guidelines
This project does not yet ship an automated test suite; prioritize adding `pytest`-based coverage for new logic. At minimum, run `python validate_merge.py` after processing or merging to confirm schema completeness, and spot-check a few chunk files for expected metadata fields. For tokenizer changes under `code/deepseek_v3_tokenizer/`, add smoke tests that instantiate the tokenizer against sample chunks. Document any manual validation steps in your pull request.

## Commit & Pull Request Guidelines
The exported archive lacks git history, so adopt Conventional Commit subjects (`type(scope): summary`) to ease release notes; keep the summary in imperative mood and <=72 characters. Each pull request should describe purpose, dataset touch-points, commands executed, and attach relevant log snippets or screenshots (for example, the `validate_merge.py` summary). Link related issues and flag any data migrations or large files that require reviewer attention. Request review once all scripts complete locally and data artifacts are regenerated or explicitly skipped.

## Data & Configuration Tips
Because several scripts assume absolute paths, replace them with configurable roots (CLI arguments or environment variables) whenever you touch those files. Avoid committing large raw data; stash them under `data/` with `.gitignore` updates if version control is added. Never upload proprietary manuals from `books/`; reference their page numbers instead. Protect any tokenizer credentials or API keys by using `.env` files excluded from version control.
