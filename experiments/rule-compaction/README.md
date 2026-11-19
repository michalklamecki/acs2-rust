# Rule compaction experiments

Registrations for the rule compaction line (`docs/RULE_COMPACTION.md`). The mechanism is
described in `docs/ARCHITECTURE.md`, "Rule compaction during learning". Nothing here has
run on the cluster, and nothing here may be submitted without the user's explicit approval.

## Purpose

Establish whether compacting the rule population during learning reduces the matching
cost of ACS2, ACS2ER, ACS2HER and ACS2VCP without losing what the agent knows or how
well it performs on held-out start-goal pairs. The work proceeds as a funnel, and each
measurement decides the next one:

1. **Shadow compaction** (`shadow-protocol.json`, registered, awaiting approval). Plain
   learning runs; at every checkpoint every operator runs on a copy of each head's
   population and records what it would remove and what that would do. Learning is
   unchanged, so the rows double as the no-compaction reference.
2. **Single reduction** (`intervention-draft.json`, draft, not approved). One reduction at
   an early, middle or late interaction count, then ordinary learning.
3. Adaptive triggering, multi-head effects and the final comparison are not registered.
   They are designed after the two measurements above.

## Frozen success measures

Recorded in `shadow-protocol.json` under `success_measures`. Changing a value after
results exist needs a new registration.

- **Cost.** Classifier perception tests in training plus compaction work in the same
  unit. Deterministic and independent of hardware. Wall time, memory and the area under
  the population-size curve are reported separately.
- **Quality.** Knowledge and held-out success (mean over the registered checkpoints and
  the final point), plus the value diagnostics.
- **"Helps"** means at least 30% fewer matching tests, including compaction work, with
  knowledge and held-out success each no more than 0.01 below the same agent without
  compaction, on mazes that were not used for tuning.

ACS2VCP with priorities performs worse than its own uniform control in this repository,
so every claim is stated against the same agent without compaction, and the uniform
controls stay in the arms.

## Tuning and evaluation split

| | Mazes | Block | Seeds |
|---|---|---|---|
| Tuning (this measurement) | MazeMA, Woods14, Maze4 | validation | 42-46 |
| Evaluation (final comparison) | Maze4, Maze5, Maze6, Maze7, Woods14, MazeMA | test | 42-61 |

Maze5, Maze6 and Maze7 are untouched by tuning and are the main evidence; claims are made
only on them. Maze4, MazeMA and Woods14 appear in the final comparison but carry no claim,
because their validation block informed the thresholds. Protocol: `maze-pairs-v1`, with
the pairs, caps, budgets and checkpoints of `experiments/maze-baselines/`; the pair
protocol is pinned by SHA-256 in both registrations.

## Files

| File | Role |
|---|---|
| `protocol.py` | Constants and builders for both registrations. |
| `register.py` | Writes `shadow-protocol.json`, `intervention-draft.json` and the single-reduction specifications. `--check` fails when a committed file differs from its generator. |
| `shadow-protocol.json` | The shadow measurement: arms, split, measures, configurations, cost, pilot and the steps before cluster use. |
| `intervention-draft.json` | The single-reduction measurement. Marked `draft_not_approved`; placeholders for operators, agents and points. |
| `test_protocol.py` | Identity, split, arms, pinned hashes, budget arithmetic and the draft's guards. |

Run `python3 -B experiments/rule-compaction/register.py` after editing `protocol.py`.

### Shadow arms

Seven arms, each a plain preset run with `--compaction-shadow`, on the three tuning mazes
and seeds 42-46: `acs2`, `acs2er`, `acs2her`, `acs2vcp_k2` and `acs2vcp_k4` at temperature
24, and their uniform controls `acs2vcp_k2_uniform` and `acs2vcp_k4_uniform` (temperature
0). The prioritized VCP arms take `--vcp-temperature 24`; the flag applies to every agent
in one invocation, so VCP arms run one agent per invocation, as the grid does.

The comparator arms A7 (GA enabled, small `u_max`) are not registered. `acs2-measure`
builds every agent from one thesis preset and has no option for either parameter, so
they are not expressible without new command-line surface. That is a separate scope
decision.

### Specification files

All files are strict JSON parsed by `acs2-measure` (`deny_unknown_fields`); the echo of
the specification appears in each row.

| File | Meaning |
|---|---|
| `shadow_all_operators.json` | Shadow set for agents with an experience memory: nine entries (exact and tolerant mature subsumption, blind and guarded CRACS subsumption, idle pruning after one and three audits, corrected dynamic thresholds, capacity 5000, mature subsumption then idle pruning), a lossless audit in observe mode, census, evidence sample 2000. |
| `shadow_without_evidence.json` | Shadow set for plain ACS2, which keeps no memory: only the operators that need no evidence (five entries), no audit, evidence sample 0. The measurement refuses a shadow with evidence for ACS2. |
| `mature_subsumption_exact_once.json` | One exact mature subsumption at interaction 1,000,000, audit observed. The base of the single-reduction files. |
| `single_reduction_early.json`, `single_reduction_middle.json`, `single_reduction_late.json` | The base at interactions 50,000, 500,000 and 1,500,000. Placeholders until the shadow gate. |
| `single_reduction_early_without_evidence.json`, `single_reduction_middle_without_evidence.json`, `single_reduction_late_without_evidence.json` | The same for plain ACS2: audit skipped, evidence sample 0. |
| `mature_subsumption_exact_growth.json` | Apply mode, exact mature subsumption when the population grows 1.5 times since the last reduction. |
| `mature_subsumption_exact_growth_without_evidence.json` | The same for plain ACS2. |
| `dynamic_thresholds_periodic.json` | The corrected thresholds of the earlier dynamic method every 50,000 interactions; a reference, not the proposal. |
| `idle_pruning_audited_cost_benefit_feedback.json` | Mature subsumption then idle pruning under a cost-benefit trigger with feedback control and an enforced audit; the ambitious candidate. |

## Cost and the pilot

The plan's expectation for the shadow measurement is at most 6 CPU-hours including the
pilot. `shadow-protocol.json` records an uncomfortable fact: the shared grid charges every
array element at its hard limit, and the archived hard limits of the same seven agents on
the three tuning mazes sum to 105 minutes per seed, 8.75 CPU-hours for five seeds, before
any shadow overhead. Expected usage is well below hard limits, but the limit sum is what
admission checks. The seed-42 pilot (21 runs, about 1.75 CPU-hours of hard limits) is
therefore mandatory and decides the limits; if the grid does not fit, the user chooses
between a higher ceiling, fewer seeds and fewer arms.

## Running a smoke test locally

Only a tiny run is acceptable on the laptop: cap at most 5, a few hundred interactions,
one seed. No release build, no grid.

```bash
cargo run -p acs2-measure -- --task maze4 --cap 5 --pool full --encoding coordinates \
  --protocol maze-pairs-v1 --evaluation-blocks validation --agents acs2er \
  --compaction-shadow experiments/rule-compaction/specifications/shadow_all_operators.json \
  --seeds 42 --targets 100,300 --out "$SCRATCH/shadow-smoke.jsonl"
```

A compacting run needs `--compaction <specification.json>` and a `_compacting` run name,
for example `--agents acs2vcp_k2_compacting --vcp-temperature 24 --compaction
experiments/rule-compaction/specifications/mature_subsumption_exact_growth.json`.

Checks that need no run:

```bash
python3 -B experiments/rule-compaction/register.py --check
python3 -B -m unittest discover -s experiments/rule-compaction -p 'test_*.py'
python3 -B -m unittest discover -s experiments/grid -p 'test_*.py'
```

## What must be approved before cluster use

Cluster submission on this line happens only with the user's explicit approval, and the
MPX line's procedures do not apply.

1. The user approves the registration and the pilot. Until then `approved` is false.
2. Check the grant before submitting: `sshare -U -u "$WCSS_USER" -A "$WCSS_ACCOUNT" -o
   RawUsage -n` against `WCSS_GRANT_CPU_HOURS`, plus the queue commitment.
3. Wire the shared grid path, which is not done: an `execution.py` with
   `--derive-manifest` for this experiment registered in
   `experiments/grid/grid_authorization.py`, and the pass-through of
   `agent_settings.compaction_shadow` to `--compaction-shadow` in
   `experiments/grid/collect.py`. The row validators already accept the eight compacting
   run names and the shadow report.
4. Build the static binary once, in a dedicated clone at a pinned clean commit.
5. Submit the seed-42 pilot, stop, and report its cost. The grid is a separate approval.

The single-reduction draft additionally needs the shadow gate: at most three operators go
forward, and the agents, operators and points are re-registered and approved.

## Mid-run readings are not ceilings

This project has repeatedly reported a flat reading as a plateau and had to withdraw it
(`CLAUDE.md`, "The failure mode this project keeps hitting"). Here that means: do not
conclude that an operator removes little, or that knowledge has stopped recovering, from
an early checkpoint or a partial run. Compare against the same agent's trajectory at the
same checkpoint, use full registered horizons, and state results only from registered
points. A knowledge value that has been flat for a long time is not converged.
