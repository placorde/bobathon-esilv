# Setup status — last updated 2026-10-12

> Support note (non-scoring). Records what's actually there, verified by running the official tests / querying the Hub API directly — not guessed.

## This machine (placorde / "databoys" workspace)

All 7 setup tests pass as of 2026-10-12:

```
tests/test_setup.py::test_python_version_is_3_12_or_newer PASSED
tests/test_setup.py::test_virtual_environment_is_active PASSED
tests/test_setup.py::test_skore_cli_is_installed PASSED
tests/test_setup.py::test_bob_skills_are_installed PASSED
tests/test_setup.py::test_skore_file_is_complete PASSED
tests/test_setup.py::test_hub_accepts_the_api_key PASSED
tests/test_setup.py::test_bob_skill_permission_is_toggled PASSED
```

`.venv` uses Python 3.13.9 (not the system 3.14.7 — a 3.13 venv was used, sidestepping the "too new, no wheels yet" risk). `.skore` points to `hub_url: https://api.skore.probabl.ai`, `workspace: databoys`. Setup is **done** — cleared for Day 1/Day 2.

## Team Hub status (checked via direct API query, read-only)

The Hub workspace `databoys` has **one existing project, confusingly also named `databoys`** (not `bobathon-esilv` as DAY1.md's example code names it — `Project(name=...)` can be any string, just keep it consistent across the team). It contains **one EstimatorReport**, pushed 2026-10-07 (5 days before this check) — confirmed with the teammate: this is real work, he got through part of the Day 1/early Day 2 material a few days ago and hasn't continued since.

| Field | Value |
| --- | --- |
| Report key | `estimator` (generic default — **not** a valid competition key like `01_dummy`/`02_ridge`) |
| Model | `Ridge` |
| RMSE train / test | 8.19 / 9.66 |
| R² train / test | 0.997 / 0.996 |
| Split | default `skore.evaluate` random 80/20 (not patient-grouped) |

**This report does not count as a finished step and should not be reused as-is:**
- No dummy-mean baseline (`01_dummy`) exists yet in this project — DAY1.md requires that *before* any real model, as the floor to beat.
- Random split leaks patient visits across train/test (same patient's rows can land on both sides) — DAY2.md step 3 flags this explicitly as misleadingly optimistic. The R²=0.997 here is likely inflated for that reason.
- The key `estimator` isn't a valid report key for a Kaggle Submission Description.

## What's actually left to do (team-wide)

1. Run G-EDA in Bob (`Explore my data in the data/ folder.`) — not done yet, independent of the stray Ridge report.
2. Push a proper `01_dummy` baseline report.
3. Redo Ridge (or reuse the idea) **with a new key** (`02_ridge`) and, once reached, patient-grouped `GroupKFold` CV rather than the default random split.
4. Agree as a team on one Hub project name going forward (`databoys` already has history — simplest is to keep using it rather than start a second `bobathon-esilv` project and fragment reports).

## Risks still worth watching

1. **Python 3.14** on the system PATH is very recent — the working `.venv` uses 3.13.9 instead, which avoided wheel-availability issues. If anyone recreates the venv, pin 3.12/3.13 explicitly (`py -3.13 -m venv --prompt bobathon .venv`).
2. **Repo under OneDrive**: large `.venv` inside a synced folder can cause slow installs or file locks. Not an issue so far, but worth remembering if a reinstall is needed.
3. **Bob permissions are per chat**: the Skill tick only applies to the current chat (SETUP.md §4). Re-tick it if a new Bob chat is opened on this repo.
