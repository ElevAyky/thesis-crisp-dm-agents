# CRISP-DM as a Finite State Machine for Steering Autonomous LLM Agents

Code, data references and analysis scripts of the master's thesis
*Process-Grounded Machine Learning: CRISP-DM as a Finite State Machine for Steering Autonomous LLM Agents*
(Johannes Steinle, M.Sc. Digital Business Engineering, Reutlingen University / Herman Hollerith Zentrum, 2026).

| Directory | Content |
|---|---|
| `src/crispfsm/` | the experimental harness: CRISP-DM state machine, gates, LLM judges, free-agent arms, scoring outside the agent |
| `scripts/` | experiment runner and analysis scripts |
| `manifests/` | dataset manifests and the stored baseline predictions |
| `tests/` | test suite (`uv run pytest`) |
| `thesis/` | fact sheets (`thesis/facts/`), the scripts that generate them (`thesis/scripts/`) and the tables, figures and appendix contents these scripts produce |

The run records (summaries, event logs, session transcripts, agent code and predictions) are in the
companion repository [`thesis-crisp-dm-runs`](https://github.com/ElevAyky/thesis-crisp-dm-runs). The scripts expect it side by side, at `../thesis-crisp-dm-runs`.
How to repeat the runs and how to regenerate every number, table and figure is described in
Appendix B of the thesis; the commands are repeated below.

## Requirements

- [uv](https://docs.astral.sh/uv/) (creates the locked Python 3.12 environment from `uv.lock` on first use)
- Claude Code 2.1.280 on the search path, logged in with a Claude subscription (the workers run as
  Claude Code sessions with Claude Opus 4.8)
- for the two post-cutoff datasets: a Kaggle account with an API token and the accepted rules of the
  competitions `playground-series-s6e7` and `playground-series-s6e9`

## Repeating the runs

Run from the root of this repository. The runner skips every run that already has a valid summary, so use
a new, empty run directory; an interrupted batch is resumed by repeating its command.

```bash
R=../thesis-crisp-dm-runs-repeat     # new, empty run directory
uvx --from kaggle python scripts/fetch_kaggle.py playground-series-s6e7
uvx --from kaggle python scripts/fetch_kaggle.py playground-series-s6e9
uv run python scripts/main_runs.py --runs-dir $R --leakage --arms naked_react
uv run python scripts/main_runs.py --runs-dir $R --leakage spurious \
    --arms naked_react
uv run python scripts/main_runs.py --runs-dir $R \
    --arms naked_react crisp_react hard_fsm soft_fsm
uv run python scripts/main_runs.py --runs-dir $R \
    --manifests-dir manifests/post_cutoff
uv run python scripts/main_runs.py --runs-dir $R --batch h1 \
    --arms hard_fsm_single hard_fsm
uv run python scripts/main_runs.py --runs-dir $R --batch h1 \
    --arms hard_fsm_single hard_fsm --manifests-dir manifests/post_cutoff
uv run python -m crispfsm.run --dataset wine --arm hard_fsm --runs-dir $R
uv run pytest
```

## Regenerating the results

With the run repository at `../thesis-crisp-dm-runs`:

```bash
R=../thesis-crisp-dm-runs
uv run python scripts/main_runs.py --runs-dir $R --aggregate-only \
    --arms naked_react crisp_react hard_fsm soft_fsm
uv run python scripts/main_runs.py --runs-dir $R \
    --manifests-dir manifests/post_cutoff --aggregate-only
uv run python scripts/analyze_main.py --runs-dir $R
uv run python scripts/analyze_main.py --input $R/_post_cutoff/main_runs.json \
    --out-dir $R/_post_cutoff/analysis
uv run python thesis/scripts/collect_facts.py --runs-dir $R
uv run python scripts/main_runs.py --runs-dir $R --batch h1 \
    --arms hard_fsm_single hard_fsm --aggregate-only
uv run python scripts/main_runs.py --runs-dir $R --batch h1 \
    --arms hard_fsm_single hard_fsm \
    --manifests-dir manifests/post_cutoff --aggregate-only
uv run python scripts/analyze_h1.py --runs-dir $R
uv run python thesis/scripts/collect_facts_h1.py --runs-dir $R
uv run python thesis/scripts/point_rule_sensitivity.py --runs-dir $R
uv run python thesis/scripts/effect_intervals.py
uv run python thesis/scripts/collect_facts_eval_return.py --runs-dir $R
uv run python thesis/scripts/make_factsheets.py
uv run python thesis/scripts/make_tables.py
uv run python thesis/scripts/make_figures.py --runs-dir $R
uv run python thesis/scripts/check_tables.py
uv run python thesis/scripts/extract_actions.py --runs-dir $R
uv run python thesis/scripts/code_process.py --runs-dir $R
uv run python thesis/scripts/validation_sample.py
uv run python thesis/scripts/coding_agreement.py
uv run python thesis/scripts/make_factsheet_kap7.py
uv run python thesis/scripts/make_figures_kap7.py
uv run python scripts/memorization_check.py --runs-dir $R
uv run python thesis/scripts/make_appendix_prompts.py --runs-dir $R
uv run python thesis/scripts/make_appendix_tables.py --runs-dir $R
```

The aggregation and analysis steps write into the run repository (`$R/_*/`); all further outputs go to
`thesis/facts/derived/`, `thesis/facts/`, `thesis/tables/`, `thesis/figs/` and `thesis/appendix/generated/`.
`check_tables.py` confirms that every decimal number, time of day and integer of 100 or more in the tables
appears in a fact sheet. Two inputs cannot be regenerated from the snapshots: the second coder's coding of
the validation sample (`thesis/facts/derived/process_coding_manual.json`) and the check of which columns the
eucalyptus models leave out (`thesis/scripts/eucalyptus_exclusions.py`), which reruns the agents' code and
needs the data copies and saved models that the run repository does not contain; its result is stored in
`thesis/facts/derived/eucalyptus_exclusions.json`. Four steps read the commit history, which this snapshot
does not contain: three of them then use the values stored in `thesis/facts/derived/`, and the fourth, a
check that the prompt files are unchanged since commit `14e99e4`, relies on the prompt fingerprint alone.

## Notes

- This repository is a cleaned snapshot of the working repository at submission, in a single commit.
  It does not contain the text of the thesis. The working repository, with its full commit history and the
  project documentation, is available to the examiners on request (Appendix A of the thesis).
- The run records name the harness commits of the working repository under which each run was executed
  (Appendix B.2).
- Comments in the code are partly in German. References of the form `docs/…` point to documents of the
  working repository; IDs of the form `F5-…` refer to the fact sheets in `thesis/facts/`.

## License

The contents of this repository are licensed under the MIT License, see [`LICENSE`](LICENSE). The datasets
are not part of this repository; they remain under the terms of their sources (OpenML, scikit-learn, Kaggle).
