# Hackathon checklist — what needs Bob vs. what doesn't

Legend: **[BOB]** = must be run/written inside Bob IDE/CLI (per competition rules §4, counts toward scoring). **[SUPPORT]** = exploration/doc/notes, safe for Claude Code or manual work, never produces graded code.

Check items off as they're done. If a `[BOB]` step fails or needs a redo, prefer fixing the prompt once rather than re-running variations — tokens are limited on both sides.

Each `[BOB]` item below has the exact text to paste into Bob's chat (or the exact UI action, when it's not a prompt).

## Step 0 — Setup

- [x] **[SUPPORT]** Join Kaggle, form team (max 4), note exact team name
- [x] **[SUPPORT]** Install Python 3.12+ on PATH (current env has 3.14.7 — verify a 3.12/3.13 venv works, see SETUP_STATUS.md)
- [x] **[SUPPORT]** Create Skore Hub workspace (skore.probabl.ai), name = Kaggle team name, invite teammates
- [x] **[BOB]** *(UI action, not a prompt)* Open repo in Bob → Permissions button (next to Mode selector) → tick **Skill** → leave Auto-approve on. Must be re-checked each new Bob chat.
- [x] **[SUPPORT]** Create `.venv`, activate it, `pip install -r requirements.txt`
- [x] **[SUPPORT]** `skore skills install all --repo probabl-ai/skills-hackathon --agent bob`
- [x] **[SUPPORT]** `python scripts/skore-agent` (sign in to Hub, writes `.skore`)
- [x] **[SUPPORT]** `python -m pytest` → must say `all passed` before Day 1 (gate test also checks the Bob Skill toggle above, so do that first)

## Step 1 — Day 1

- [x] **[BOB]** Prompt: `Explore my data in the data/ folder.` → runs G-EDA, produces `data/eda.py`, `data/eda.md`, `data/eda_<table>.html`
- [ ] **[SUPPORT]** Answer Day1 workbook Q1–6 from the EDA output (draft answers already in `EDA_FINDINGS.md`, verify once G-EDA has run for real)
- [x] **[BOB]** Prompt: `Run a dummy regressor from sklearn to act as my baseline model. Push to the hub with the name '01_dummy' as the report name to the project 'bobathon-esilv'.` — done 2026-10-12 16:03, pushed to project `databoys` key `01_dummy`
- [ ] **[SUPPORT]** Answer the 3 workbook questions on the dummy report (RMSE vs. target std, error-plot shape, "real improvement" threshold)

## Step 2 — Day 2

- [x] **[BOB]** Prompt: `Run a linear model and push the report to the hub. Please compare the results to the dummy model.` — done 2026-10-13 10:28, test RMSE 10.40 vs dummy 16.48 (-36.9%)
- [x] **[BOB]** Prompt: `Tune the Ridge alpha (try 0.1, 1.0, 10.0), keep the best one, push it to the hub as '02_ridge', then fit it on all training visits and write submissions/02_ridge.csv for the Kaggle test set.` — done 2026-10-13, alpha had ~no effect (test RMSE 10.40 both times), submissions/02_ridge.csv exists
- [ ] **[SUPPORT]** Upload `02_ridge.csv` to Kaggle with the Hub report URL in the Submission Description — **first real submission** (manual step, user must do this on kaggle.com)
- [x] **[BOB]** Prompt: `Re-evaluate the dummy and ridge models using patient-grouped cross-validation (GroupKFold on patient_id) instead of the default random split, and push the comparison report to the hub with a new key.` — done 2026-10-13 11:38-11:39, keys `03_dummy_grouped`/`03_ridge_grouped`. RMSE barely moved (16.50 vs 16.48 dummy, 10.45 vs 10.40 ridge) — grouping didn't change much here
- [x] **[BOB]** Prompt: `Now evaluate a HistGradientBoostingRegressor with the same grouped CV, keeping missing values as-is (no imputation), and push it to the hub with a new key.` — done 2026-10-13 13:46, key `04_hgbr_grouped`: MAE 6.10 (vs Ridge 8.39, -27%), R² 0.78, stable (MAE std 0.105)
- [ ] **[BOB]** Prompt: `Try skrub's tabular_pipeline("regressor") so categorical columns like gene and cohort are included, using the same grouped CV, and push it to the hub with a new key.`
- [ ] **[BOB]** Prompt: `Rebuild the best model using skrub DataOps, with groups baked into the graph via mark_as_X(cv=GroupKFold(...), split_kwargs={"groups": groups}), and push it to the hub with a new key.`
- [ ] **[BOB]** Prompt: `Fit the best model on all training visits, predict on X_test, and write submissions/<key>.csv matching the report key we just used.`
- [ ] **[SUPPORT]** Upload final CSV to Kaggle with its report URL — designate as the team's scored leaderboard entry

## Validity reminders (every `[BOB]` + submit pair)

- New Hub `EstimatorReport` key per Kaggle file, never reused, never `eda` for a model
- CSV header `Index,target`, filename = report key
- Submission Description **must** contain the live `https://skore.probabl.ai/…` URL
- Only one submission designated as the team's scored entry
