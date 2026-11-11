# RULE_COMPACTION.md — goal, constraints and plan of the rule compaction line

This file defines the rule compaction line. It states what the thesis is, which
measurement rules bind every experiment on this line, and the order of the work. How the
compaction code is built is recorded in `docs/ARCHITECTURE.md`, in the "Rule compaction
(CRACS)" section, and ACS2VCP is described in "Value consistency prioritization with
independent hindsight heads". The MPX and trajectory utility lines in `CLAUDE.md` are
separate work and their procedures do not apply here.

## Thesis

- **Title (Polish, official):** Rozszerzenie modelu ACS2VCP o adaptacyjne kompaktowanie
  populacji reguł
- **Title (English):** Extending ACS2VCP with Adaptive Rule Population Compaction

### Goal and scope

Develop an extension of ACS2VCP that limits the size of the rule population and the
computational cost while preserving the quality of the environment model and the agent's
performance. The scope covers an implementation in Rust and a comparison with ACS2,
ACS2ER, ACS2HER and baseline ACS2VCP.

### Engineering aspect

- A multi-head ACS2VCP in Rust with a shared experience memory and a priority update
  mechanism.
- A rule compaction module.
- Parity tests against the reference implementation.
- Tools that measure runtime and memory use.
- A check of whether the per-head computation can run in parallel.

### Research aspect

- Adapt rule compaction methods to ACS2VCP.
- Design a criterion that triggers compaction during learning.
- Study how population reduction affects each head's predictions, the experience
  priorities derived from them, and the agent's performance.
- Characterise the trade-off between computational cost, knowledge retention and the
  diversity of the model's heads.

### Tasks

- Analyse ACS2VCP and rule compaction methods.
- Implement in Rust a variant with adaptive population reduction.
- Compare it with ACS2, ACS2ER, ACS2HER and ACS2VCP at a controlled number of interactions.
  Measure compaction cost, memory and runtime separately.
- Write the documentation and the experiment configurations.
- Analyse the range in which population reduction preserves the quality of the
  environment model.

## Rules that bind every experiment on this line

- **Origin of the idea.** Compaction follows from the limitations and future work named in
  section 3.7 and the conclusion of Zemła's thesis, the base work that defines ACS2VCP.
  The name ACS2VCP follows that work.
- **Heads are independent ACS2 instances.** They are not a neural actor and critic and must
  not be described or modelled as one. In the code each head owns its population, random
  source and counters (`acs2-agents/src/acs2vcp/`).
- **Equal budgets are counted in environment interactions summed over all heads.** Equal
  episode counts do not give equal interaction counts. Every comparison also reports the
  number of updates and the compute time.
- **Compaction is not only a memory optimisation.** It reduces the population while
  preserving three things: coverage of the relevant states, the predicted effects of
  actions, and useful value estimates.
- **Every reduction is measured before and after.** Changing a population can change the
  experience priorities in ACS2VCP and with them the rest of learning. Prediction quality is
  therefore measured before and after each reduction.

## Plan

1. **Analysis of the model and of compaction methods.** Study Zemła's thesis, the ACS2VCP
   implementation and CRACS. Describe how the heads' rule populations depend on the shared
   experience memory and on the sampling priorities. Decide which rule properties must
   survive a reduction.
2. **Reference implementation in Rust.** Provide ACS2, ACS2ER, ACS2HER and ACS2VCP behind a
   common environment interface, with configuration and seeds recorded. Verify rule
   matching, value updates, priorities and multi-head operation on small examples and
   against the available reference code.
3. **Profiling ACS2VCP.** Measure the time of rule matching and updates, the time spent in
   the experience memory, and memory use, for several population sizes and head counts.
   Identify the operations whose cost actually grows with the population. Those results set
   the optimisation target.
4. **Compaction as a separate operation.** Implement the chosen reduction method and study
   it first on saved populations. On a held-out set of training transitions, compare
   coverage, predicted effects and action values before and after the reduction. Define a
   rejection condition for a reduction that degrades these properties too far.
5. **Adaptive triggering.** Design a criterion that uses the population size and the
   estimated compute cost, with a check on the loss of prediction quality. Compare it with no
   compaction and with compaction every fixed number of interactions. After a population
   changes, keep the estimates that the experience priorities depend on up to date.
6. **Effect on how the heads work together.** Check that compacting each head on its own
   does not remove their useful diversity. Measure the change in prediction disagreement and
   in the priority distribution. Evaluate parallel execution separately, to tell an
   algorithmic gain from an implementation speed-up.
7. **Comparative experiments.** Compare the full extension with the baseline models, and its
   variants across head counts and reduction thresholds. Equalise the total interactions of
   all heads, use many seeds and measure compaction cost separately. Tune parameters on a
   separate task set before the final evaluation.
8. **Analysis and write-up.** Tabulate agent performance, rule counts, coverage, prediction
   errors, time and memory. Determine the range in which reduction pays off without excessive
   loss of quality, and report negative results as well. Prepare documentation and a package
   that reproduces the experiments.

## Starting point in this repository

- `acs2-agents/src/acs2/`, `acs2er/`, `acs2her/` and `acs2vcp/` already hold the four
  baseline agents. ACS2VCP has the registered presets `acs2vcp_k2`, `acs2vcp_k4` and their
  `_uniform` variants.
- `acs2-core/src/compaction/` holds CRACS as a one-shot, non-destructive population
  operation with its own `CompactionConfiguration`. Nothing on an existing learning path
  calls it yet.
- The cluster account settings are per user and live in `.env`; see `.env.example` and the
  cluster section of `README.md`.
