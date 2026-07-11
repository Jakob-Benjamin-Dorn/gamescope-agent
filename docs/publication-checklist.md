# Publication checklist

Before making the repository public:

- [ ] Replace `Jakob` in `pyproject.toml` and `LICENSE` with the preferred public name.
- [ ] Review `NOTICE.md` and keep the course-origin disclosure.
- [ ] Copy `.env.example` to `.env` locally; never commit `.env`.
- [ ] Run `python -m ruff check .`.
- [ ] Run `python -m pytest`.
- [ ] Run `gamescope ingest` in a clean virtual environment.
- [ ] Run at least one local-answer and one web-fallback example.
- [ ] Add one screenshot or terminal trace to the README if desired.
- [ ] Create the GitHub repository without uploading the original course notebooks.
