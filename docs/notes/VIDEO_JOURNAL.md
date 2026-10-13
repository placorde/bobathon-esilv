# Logbook: material for the final video (5 min max)

> Kept as we go (Claude Code support, non-scoring). One entry = date, what happened, and why it matters for the video.
> **Final script budget: about 650–750 spoken words in total.** Keep the 1 or 2 strongest points per dimension (about 160–190 words each), not an answer to every sub-question. This file collects material; the selection happens at the end.
> Each entry is tagged with the question(s) it answers (e.g. `[D1.2]`).
> Step progress is tracked only in [HACKATHON_CHECKLIST.md](HACKATHON_CHECKLIST.md); this file records what happened and what it taught us.

## Dimension 1: Challenge & Learning

1. How would you describe the main challenge you faced during the hackathon?
2. What made the data used in this hackathon interesting or challenging from a learning perspective?
3. What new skills, knowledge, or competencies did you develop or strengthen during the hackathon?
4. How did working on this real-world use case contribute to your understanding of machine learning?
5. What was the most important thing you learned from working on this challenge?
6. How did your approach to solving the ML problem evolve throughout the hackathon?

**Entries**
- **2026-10-12** `[D1.2][D1.5]`: the true OFF target is **monotonic and almost linear** over time for each patient (median correlation of 0.98 with disease duration). We are not predicting isolated visits; we are rebuilding a **trajectory per patient**. *(Source: [EDA_FINDINGS.md](EDA_FINDINGS.md))*
- **2026-10-12** `[D1.2][D1.4]`: measured `off` **underestimates** the target by about 5 points, with a bias that depends on the **rater** (−10.5 to +2.2) and on the **time since intake**. These are exactly the real-world biases described in CONTEXT.md (subjectivity, drug timing): a concrete case of a "debiased" target.
- **2026-10-12** `[D1.2][D1.3]`: missing values are a **clinical signal**: no LEDD means not treated yet, which means an early stage. Lesson: do not blindly fill gaps with the median.
- *To follow* `[D1.6]`: log each change of approach (dummy → Ridge → grouped CV → HGBR → skrub → per-patient features?) with the RMSE gain at each step.

## Dimension 2: IBM Bob Usage

1. How did you use IBM Bob throughout the different stages of the hackathon?
2. What types of tasks did you use IBM Bob for, and how did it support your team's work?
3. Which features or capabilities of IBM Bob did you find most useful, and why?
4. Can you describe a situation where IBM Bob significantly helped you or your team?
5. In which parts of the ML workflow did IBM Bob provide the greatest added value?
6. How did using IBM Bob influence the way you approached or solved the challenge?
7. What would you have done differently if IBM Bob had not been available?
8. How would you describe the role of IBM Bob in your overall hackathon experience?

**Entries**
- **2026-10-12** `[D2.3]`: the repo ships a Bob rule (`.bob/rules/AGENTS.md`) that **forces Bob to run the setup tests before any action**. A concrete example of the "Rules" feature (video 4 of the playlist, see [BOB_VIDEO_GUIDE.md](../BOB_VIDEO_GUIDE.md)). Worth filming: Bob refusing to start while the setup is incomplete.
- *To fill* `[D2.1][D2.4]`: first G-EDA run (`explore-ml-data`). What did Bob produce, and do its numbers match [EDA_FINDINGS.md](EDA_FINDINGS.md)? (A match or a mismatch both make a good anecdote.)
- *To fill* `[D2.3][D2.5]`: which skills were used (out of the 14 in `.bob/skills/`), modes, permissions, and at which stage (EDA, pipeline, evaluation, iteration).
- *To fill* `[D2.6]`: a moment when a suggestion from Bob changed the approach, or a moment when it had to be corrected.

## Dimension 3: Technical Continuity

1. How did your use of the technical environment and tools evolve during the hackathon?
2. Which tools, techniques, or components developed during the hackathon do you plan to continue using in your ML project?
3. How have you reused, or how do you plan to reuse, the technical setup after the hackathon?
4. What happened to your workflow when you worked without IBM Bob, and what did you notice compared with using it?
5. Which aspects of the technical setup would you keep, change, or improve for your future ML projects?
6. How has the hackathon influenced the way you approach future ML projects?

**Entries**
- **2026-10-12** `[D3.4]`: how the work is split is itself material for this question. All scored code goes through Bob + skore (competition rules). Another assistant (Claude Code) was used **for non-scoring work only**: reading the docs, checking the setup and running descriptive stats. Compare the two workflows honestly.
- *To fill* `[D3.2][D3.5]`: what we keep (patient-grouped CV, traceable skore reports, one report key = one CSV, skrub DataOps) and what we would change.

| key | model / change | RMSE (random split) | RMSE (GroupKFold) | Kaggle public | Hub URL |
| --- | --- | --- | --- | --- | --- |
| 01_dummy | mean | ≈16.5 expected | | | |

## Dimension 4: Overall Experience

1. How would you describe your overall experience of the hackathon?
2. How would you describe the organization and collaboration within your group?
3. How did your team organize the work and divide responsibilities during the challenge?
4. How did the support provided by the hackathon team contribute to your progress and experience?
5. What aspects of the hackathon worked particularly well from your perspective?
6. What difficulties or obstacles did you encounter during the hackathon, and how did you address them?
7. What could be improved to make the hackathon more effective or valuable for future participants?
8. What is the main takeaway you will remember from this hackathon?

**Entries**
- **2026-10-12** `[D4.5]`: the lab docs are well guided (SETUP → DAY1 → DAY2, with workbook questions at each step), and the setup tests tell you which step to redo.
- **2026-10-12** `[D4.7]`: small doc inconsistencies found: a `time_since_diagnosis` column announced but absent, and a duplicated paragraph in the README. Possible material for the "what could be improved" question.
- *To fill* `[D4.2][D4.3]`: who does what in the team, and how the Hub workspace is shared.

## Obstacles and pivots (feeds D1.1, D1.6, D4.6)

| date | obstacle / pivot | how we solved it | questions |
| --- | --- | --- | --- |
| 2026-10-12 | incomplete setup: no `.venv`, `.skore` or skills; Python 3.14 and repo under OneDrive (risks) | see [SETUP_STATUS.md](SETUP_STATUS.md) | D4.6, D3.1 |
| 2026-10-12 | DAY1.md announces a `time_since_diagnosis` column in test that does not exist | compute `age - age_at_diagnosis` on both sides | D4.7 |
