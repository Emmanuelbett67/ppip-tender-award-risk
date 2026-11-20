# AI use disclosure

Course: DSA 8401 Applied Machine Learning, Strathmore University (iLabAfrica)
Student: Emmanuel Bett (131611)

This disclosure is made in accordance with the course policy on AI assistance.

## Tools used

| Tool | Provider | Used for |
|---|---|---|
| Claude Code (Claude models) | Anthropic | Coding assistant in the terminal: drafting and refactoring Python code, writing documentation, debugging |

No other generative-AI tools were used. No AI tool was given access to the raw PPIP database beyond what runs locally on my machine.

## What the AI assisted with

- **Repository engineering.** Converting the research notebook into the scripted, reproducible pipeline in `src/tender_risk/` and `scripts/`; writing the FastAPI service, tests, Dockerfile, `docker-compose.yml`, CI workflow and Makefile.
- **Code drafting in the notebook.** Drafting and debugging analysis code (data parsing, feature engineering, model training loops, statistical tests, plots) from my specifications of what each section should compute.
- **Documentation.** Drafting this README, the final report and inline docstrings from my notes and the notebook results, which I then edited.

## What I did myself

- Chose the problem, obtained the PPIP data export and wrote the proposal.
- Took every study-design decision: the scoring point (information available at the closing date), the exclusion of post-closing fields such as bidder counts, the outcome taxonomy and the three outcome definitions, forward-chaining temporal validation, the choice of model families, the fairness criteria and the model-selection rule.
- Ran every experiment, checked outputs against the raw data, and interpreted the results. All numbers in the report were produced by code in this repository, not by an AI tool.
- Reviewed, tested and edited all AI-drafted code before committing it. I am responsible for its correctness.

## How to verify

Everything the AI helped write is in this public repository and can be re-run from scratch (`README.md`, "Reproduce"). The research notebook that records the analysis step by step is in `notebooks/`.
