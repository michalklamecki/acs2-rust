# RULE_COMPACTION.md — goal, constraints and plan of the rule compaction line

This file defines the rule compaction line. It states what the thesis is, which
measurement rules bind every experiment on this line, and the order of the work. How the
compaction code is built is recorded in `docs/ARCHITECTURE.md`, in the "Rule compaction
during learning" section (the one-shot CRACS operation is in "Rule compaction (CRACS)"),
and ACS2VCP is described in "Value consistency prioritization with independent hindsight
heads". Registered measurements and their status are in
`experiments/rule-compaction/README.md`. The MPX and trajectory utility lines in `CLAUDE.md` are
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

## Live state

### What exists

Everything below is implemented and tested locally. Nothing has run on the cluster.

- **Operators** (`acs2-core/src/compaction/`), each a pure function from a population to a
  `ReductionOutcome` with one verdict per rule:
  - A1 guarded CRACS subsumption. Unguarded, it is the one-shot operation of
    "Rule compaction (CRACS)": knowledge is preserved exactly, values are not looked at.
  - A2 mature backward subsumption. The subsumer must satisfy ACS2's own insertion
    criterion (reliable, experience above `theta_exp`, unmarked, same action and effect)
    and include the candidate's condition. Knowledge is preserved exactly, and a removed rule
    would be rejected today if it were inserted, so it does not return while the subsumer
    stays mature.
  - A3 value guard with tolerance `delta`, an overlay on A1 and A2: a change-anticipating
    candidate goes only to a change-anticipating survivor whose fitness is at least the
    candidate's minus `delta`. At `delta = 0` every head's `Q(s, a)` is identical before and
    after, so the variance between heads and the VCP priorities are identical too. Rules are
    resolved most general first, so the tolerance does not accumulate along a chain.
  - A4 idle pruning on memory evidence. A rule is idle when, on the sampled transitions, it
    is neither the best reliable anticipator of an action set, nor a value carrier, nor part
    of an action set without a reliable anticipator, and has experience of at least
    `min_experience`. It is removed after `idle_audits` consecutive idle audits. No guarantee beyond measurement.
  - A5 dynamic thresholds, the corrected version of the earlier dynamic method: above a size
    threshold, remove rules with experience at least the experience threshold and quality
    below the quality threshold. A reference arm.
  - A6 hard capacity with eviction of the least useful rules, value carriers last.
- **Audit** (`audit.rs`): before and after on a sample of memory transitions. Coverage by a
  reliable anticipation, split into state-changing and no-change transitions; maximum and
  mean change of `Q`; fraction of states whose greedy action set changes. Thresholds give a
  verdict; `Skip`, `Observe` and `Enforce` decide whether it only records or also rejects.
- **Census** (`census.rs`): rules, numerosity, reliable, marked, change-anticipating,
  goal-specific (specified in a goal attribute), value carriers, experience quartiles and
  specificity.
- **Removal ledger** (`ledger.rs`): the share of removed rules that reappear, the cheapest
  internal signal of harm.
- **Triggers** (`trigger.rs`): C0 `Never` and `Once`, C1 `Periodic`, C2 `Growth`, C3
  `CostBenefit` (predicted saved tests against the predicted cost of a pass, both in
  classifier tests), C4 feedback control of strength and holdoff from the yield, the
  guard's rejections and the restored share, C5 phase modulation from rule creation rate and
  coverage. Strength is the triple `(value_tolerance, maturity_experience, idle_audits)`.
- **Controller** (`controller.rs`): modes `Off`, `Shadow` and `Apply`. Pipeline: trigger
  decision, evidence sample, operators in order, optional audit, optional candidate check,
  accept or reject, apply. Rejection is free because the candidate is a new population.
  Every reduction leaves a `ReductionRecord`.
- **Clocks for several heads.** Per head: each head owns a controller and fires on its own
  interaction count. Synchronous: one policy on the summed signals fires every head together.
  ACS2VCP additionally runs an acceptance audit (the mean VCP acceptance and variance on the
  evidence, before and after, with an optional veto) and a disagreement probe.
- **Compacting variants and run names.** `acs2_compacting`, `acs2er_compacting`,
  `acs2er_episodic_compacting`, `acs2her_compacting`, `acs2vcp_k2_compacting`,
  `acs2vcp_k4_compacting`, `acs2vcp_k2_uniform_compacting`, `acs2vcp_k4_uniform_compacting`.
  They compact at episode ends. Plain ACS2 keeps no memory and refuses operators and audits
  that need evidence.
- **Measurement.** `acs2-measure --compaction <specification.json>` is required for exactly
  the compacting run names. `--compaction-shadow <specification.json>` works with any run
  name: at every checkpoint each shadow entry runs on a copy of each head's population with
  strided evidence from the agent's own memory and no random draws. Rows gain the optional
  keys `compaction` and `compaction_shadow`; existing rows are unchanged. The row validators in
  `experiments/grid/` accept the eight names, validating a compacting row like its plain
  family plus the report.
- **Specifications and registrations** in `experiments/rule-compaction/`.

### What has been run

Nothing on the cluster. Local checks were unit tests and tiny smoke runs (one maze, a few
hundred interactions). No scientific result exists yet, so no claim about how much a
population can be reduced, or what it costs, is made anywhere.

### Next step

The shadow measurement (`experiments/rule-compaction/shadow-protocol.json`) is approved by the
user (2026-11-19: the seed-42 pilot, then the grid of seeds 42-46, with a CPU-hour ceiling above
the plan's 6; `approval.json`) and wired to the shared grid path (`execution.py`,
`collect.py` pass-through of `compaction_shadow`, hash verified on the node). Registered hard
limits are `max(2 x archived, archived + 30)` minutes per run; the pilot reserves 882 minutes
and the grid 3,990, so the ceiling is 82 CPU-hours. The next step is the seed-42 pilot from the
cluster clone, as described in `experiments/rule-compaction/README.md`, "Submitting the shadow
measurement". Its gate decides which at most three operators go forward to the single-reduction
measurement, which exists only as a draft.

### Known open questions

- The audit thresholds in the example specifications are placeholders, and so are the C3
  priors (prior yield, floor, margin); both wait for the shadow data.
- The trigger's learning-share matching cost is a midpoint estimate: learning formations are
  charged at the mean of the episode's start and end population sizes.
- Synchronous feedback aggregates per-head feedback (sums, any rejection, the largest
  restored share). Other aggregations are possible and not compared.
- The idle tracker also advances in `Shadow` mode. A rule that reaches the required streak
  is dropped from the tracker as if it had been removed, but it stays in the population and
  starts again from zero at the next audit, so shadow streaks differ from what an applied run
  would see. Whether that matters for reading the shadow data is not settled.
- `acs2er_one_update` has no compacting wrapper.
- The compacting wrappers do not implement `Checkpointed`; a compacting run cannot be
  resumed from a checkpoint.
- Compaction wall time cannot be separated from `train_episode`, because the reduction runs
  inside it. The deterministic work counters are the cost measure; wall time is reported as a
  total.
- Marks block mature subsumption: `does_subsume` requires an unmarked subsumer, and a general
  change-anticipating rule in a small test usually carries marks, so A2 may remove much less
  than A1. Whether that is the case is an empirical question for the shadow measurement.
- The comparator arms A7 (GA, small `u_max`) cannot be expressed on the measurement command
  line and are not registered.

### Other

- The cluster account settings are per user and live in `.env`; see `.env.example` and the
  cluster section of `README.md`.
