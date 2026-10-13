# Bob tutorial playlist — recap

Source: [IBM Bob — "Your Agentic AI Coding Partner"](https://www.youtube.com/playlist?list=PL-aWPAFzESivt1wnV8_OSmp8t1335ahN6) (13 videos, 1 unavailable/hidden — 12 watchable below). Total runtime ≈ 53 min.

## Series A — "Bob Basics" (start here, ~18 min total)

Hands-on walkthrough of the core workflow, in order.

| # | Video | Duration | What it covers |
|---|---|---|---|
| 1 | [Understand the codebase, draw an architecture diagram, start the application](https://www.youtube.com/watch?v=JKnxSiTlvs8) | 1:57 | First steps with Bob: ask it to explain an unfamiliar codebase, generate an architecture diagram, and start the app — no manual digging through files. |
| 2 | [Create an implementation plan, build your first feature, understand the context window](https://www.youtube.com/watch?v=Sa6n9hmC1tk) | 4:49 | How to get Bob to draft an implementation plan for a new feature before coding, then implement it, with a look at how the context window works. |
| 3 | [Reviewing Code Changes, running multiple agents in parallel](https://www.youtube.com/watch?v=X_-sl74pzbs) | 3:39 | Using Bob to review a diff and fix findings, plus running several Bob agents in parallel on different tasks. |
| 4 | [Use Rules, Modes and Skills to customize Bob's behaviour](https://www.youtube.com/watch?v=qdAozfL1mXw) | 7:28 | Rules, Modes, and Skills — how to tailor Bob's behaviour to your/your team's conventions (chapters: "Customizing AI agents", "Rules, modes, and skills"). |

**Relevance to the hackathon**: this is the series to watch before Day 1 — #1 and #2 map directly to exploring `data/`/the repo and building your first (dummy) model; #4 (Skills) matters because the lab setup explicitly requires enabling Bob's **Skill** permission to use the skore skills (see [SETUP.md](SETUP.md)). This repo already ships a live example of a **Rule**: [`.bob/rules/AGENTS.md`](../.bob/rules/AGENTS.md) forces Bob to run the setup tests before doing anything else in every conversation.

## Series B — Product demos (deeper dives, ~35 min total)

More polished demo-style videos, each focused on one capability.

| # | Video | Duration | What it covers |
|---|---|---|---|
| 5 | [Onboarding: What is this application's purpose?](https://www.youtube.com/watch?v=FUwWnVp7yz8) | 2:11 | Asking Bob to summarize what an unfamiliar app/repo is for. |
| 6 | [Onboarding: How do I start this application?](https://www.youtube.com/watch?v=fJvBbPTK778) | 2:28 | Bob figuring out setup/run instructions instead of digging through config files. |
| 7 | [Onboarding: Explain the architecture of this application](https://www.youtube.com/watch?v=fO43F8LlOSE) | 4:22 | Turning manual code exploration of the architecture into a quick explanation from Bob. |
| 8 | [Proactive secure coding](https://www.youtube.com/watch?v=VynsWjLQhTE) | 3:29 | Bob scanning for security vulnerabilities live in the IDE instead of waiting for a downstream scan. |
| 9 | [Architect a new feature](https://www.youtube.com/watch?v=lBa6-rVq6eE) | 5:52 | Bob planning/reasoning through a design before writing code (example: adding a reading-time indicator to an article feed). |
| 10 | [Implement a new feature](https://www.youtube.com/watch?v=9KEw_glsLPw) | 9:30 | Taking an idea from concept to fully implemented feature with Bob, end to end. |
| 11 | [Shift-left code review](https://www.youtube.com/watch?v=xzxnH305SEE) | 5:03 | Bob as a "first-pass reviewer" inside the IDE, before you'd open a pull request. |
| 12 | [World first: Literate Coding](https://www.youtube.com/watch?v=sBWzf1QiXgQ) | 2:21 | "Literate Coding" mode — blending natural language and code in the same flow instead of switching between docs and editor. |

**Relevance to the hackathon**: #9/#10 ("Architect"/"Implement a new feature") map to Day 2's model-building loop (Ridge → HistGradientBoosting → skrub); #11 (Shift-left review) is useful before each Kaggle submission; #8 (secure coding) is less relevant here since there's no sensitive data path in this challenge.

## Suggested viewing order for this hackathon

1. Videos 1–4 (Bob Basics) — before Setup/Day 1, to know the basic moves.
2. Video 9 or 10 (Architect / Implement a feature) — right before building your first real model on Day 2.
3. Video 11 (Shift-left review) — right before your first Kaggle submission.
4. The rest (5–8, 12) are optional/nice-to-have, watch if time allows.

*Note for the final video deliverable (Dim 2 — "IBM Bob Usage"): if a feature from one of these tutorials (parallel agents, rules/skills, literate coding, shift-left review) turns out to be something your team actually used and found valuable, that's good concrete material for the 5-min recap video.*
