# Rule compaction experiments

Registrations for the rule compaction line (`docs/RULE_COMPACTION.md`). The mechanism is
described in `docs/ARCHITECTURE.md`, "Rule compaction during learning". Nothing here has
run on the cluster yet. The user approved the shadow measurement (the seed-42 pilot and then the
grid) on 2026-11-19, with a ceiling above the plan's 6 CPU-hours (`approval.json`); nothing else
here may be submitted without the user's explicit approval.

## Purpose

Establish whether compacting the rule population during learning reduces the matching
cost of ACS2, ACS2ER, ACS2HER and ACS2VCP without losing what the agent knows or how
well it performs on held-out start-goal pairs. The work proceeds as a funnel, and each
measurement decides the next one:

1. **Shadow compaction** (`shadow-protocol.json`, registered and approved, wired to the shared grid path, not yet run). Plain
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
| `register.py` | Writes `shadow-protocol.json`, `intervention-draft.json` and the single-reduction specifications. `--check` fails when a committed file differs from its generator or `approval.json` does not match `shadow-protocol.json`. |
| `shadow-protocol.json` | The shadow measurement: arms, split, measures, configurations with their limits, cost, execution, pilot and the steps before cluster use. |
| `approval.json` | The user's approval of 2026-11-19 for the shadow pilot and grid and the CPU-hour ceiling. Carries the SHA-256 of `shadow-protocol.json`; `register.py --check` fails when they differ, and any change to the registration needs a fresh approval. |
| `execution.py` | `--derive-manifest` for the shared grid path (called by `experiments/grid/grid_authorization.py`) and `--print-manifest pilot\|grid` that prints the manifest to submit. Both refuse an unapproved or altered registration, a changed pair protocol or a changed shadow specification. |
| `test_execution.py` | Derivation, approval, shared-path authorization and a mocked end-to-end submission of the pilot and the grid. |
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

## Cost, limits and the pilot

The plan expected at most 6 CPU-hours. The archived hard limits of the same seven agents on
the three tuning mazes already sum to 105 minutes per seed (8.75 CPU-hours for five seeds)
before any shadow overhead, and the shared grid charges every array element at the largest
limit of its array. The user therefore approved a higher ceiling on 2026-11-19.

- **Limit per run** (registered in `configurations[].limits_minutes`):
  `max(2 x archived limit, archived limit + 30 minutes)`. Shadow overhead is unknown and a
  timeout wastes a run; reserving wall time is cheap against the 500 CPU-hour grant.
- **Array limit** is the largest limit among its elements: 34 minutes for MazeMA, 38 for
  Woods14, 42 for Maze4 (all from the prioritized k=4 arms).
- **Pilot**: one array of 21 runs (7 arms x 3 mazes, seed 42), concurrency 21, 21 x 42 = 882
  minutes of hard reservation.
- **Grid**: seeds 42-46, one array of 35 runs per maze, chained with `afterok`, concurrency 35:
  35 x (34 + 38 + 42) = 3,990 minutes. Seed 42 is run again so the grid is one self-contained
  set of 105 runs; its learning fields must equal the pilot's.
- **Ceiling**: 882 + 3,990 = 4,872 minutes = 81.2 CPU-hours, registered as 82. The grid
  manifest carries the whole pilot reservation (52,920 CPU-seconds) as its pilot allocation, so
  pilot and grid together cannot exceed the ceiling. These are reservations: the admission
  bound, not the expected use. Real consumption is the CPU time the runs take.
- **Memory** is 1024 MiB per run (512 MiB in the archived registrations), because the shadow
  holds copies of every head population and an evidence sample.

The pilot reports the measured cost beside these limits. If a run hits its limit, or the
measured cost shows the limits are far off, the limits and the ceiling are re-registered and
re-approved; nothing is reduced silently.

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

## Submitting the shadow measurement

The registration and the pilot-then-grid plan are approved (`approval.json`). Only the
orchestrating session submits, from this line's own clone on the cluster, never from the MPX
clone, and never rebuilding it.

Before the first submission:

1. Commit and push everything that the measurement needs, then check out that commit in the
   cluster clone `~/acs2-rust`. The tree must be clean: the node refuses a dirty clone and a
   clone whose `HEAD` differs from the plan.
2. Build the static musl binary once in that clone at that commit
   (`target/x86_64-unknown-linux-musl/release/acs2-measure`). Its SHA-256 and the commit are
   recorded in `plan.json` of each result root when the manifest is prepared; they are not
   registered in advance.
3. The clone's `.env` (or the environment) holds `WCSS_USER`, `WCSS_ACCOUNT` and
   `WCSS_GRANT_CPU_HOURS`; `experiments/grid/cluster_account.py` reads them. Check the grant:
   `sshare -U -u "$WCSS_USER" -A "$WCSS_ACCOUNT" -o RawUsage -n` against
   `WCSS_GRANT_CPU_HOURS`, plus the queue commitment. `submit.py` repeats the check before every
   array and refuses when the shared reservation does not fit.

Result roots live outside every checkout, under `~/compaction_runs/`. A root that already
exists is refused; investigate it instead of resubmitting.

Pilot (login node, Python 3.9):

```bash
cd ~/acs2-rust
mkdir -p ~/compaction_runs
python3 -B experiments/rule-compaction/execution.py --print-manifest pilot --repo ~/acs2-rust \
  > ~/compaction_runs/shadow-pilot-manifest.json
python3 -B experiments/grid/submit.py --root ~/compaction_runs/shadow-pilot-2 \
  --manifest ~/compaction_runs/shadow-pilot-manifest.json --repo ~/acs2-rust
```

The first attempt wrote `~/compaction_runs/shadow-pilot`, where every element stopped because the
registered specification paths were relative to the experiment directory and `collect.py` resolves
them against the repository root; it consumed about 21 CPU-seconds. That root stays as it is, and
the next attempt uses `shadow-pilot-2`. Only the completed pilot is charged to the grid as its
pilot allocation; the stopped attempt counts by the CPU time it consumed, which the 2,880
CPU-seconds between the summed reservations (292,320) and the ceiling (295,200) cover.

Wait for all 21 runs, pull the root without deletion, verify the rows and report the measured
cost. Then the grid, which needs the approval file:

```bash
python3 -B experiments/rule-compaction/execution.py --print-manifest grid --repo ~/acs2-rust \
  > ~/compaction_runs/shadow-grid-manifest.json
python3 -B experiments/grid/submit.py --root ~/compaction_runs/shadow-grid \
  --manifest ~/compaction_runs/shadow-grid-manifest.json --repo ~/acs2-rust \
  --approval ~/acs2-rust/experiments/rule-compaction/approval.json
```

What the shared path checks:

- **At submission**: the manifest equals the one `execution.py` derives from the committed,
  approved registration (arms, mazes, seeds, limits, specification paths and hashes,
  temperatures, concurrency, ceiling); no unregistered field; the approval file carries the
  registration hash; the pair protocol and every specification file still have their
  registered SHA-256; the clone is clean; pilot + used + reserved + new limits fit the ceiling
  and the shared grant.
- **On the node**: the plan bytes, the clone's `HEAD` and cleanliness, the allocation identity
  and remaining time, the CPU model, the binary SHA-256 and the preset.
  `collect.py` resolves `agent_settings[run].compaction_shadow` (registered relative to the
  repository root, for example `experiments/rule-compaction/specifications/shadow_all_operators.json`) inside the clone, refuses an
  absolute path or one that leaves the clone, verifies the specification's SHA-256 against
  `compaction_shadow_sha256` before the process starts, and passes `--compaction-shadow <path>`.
  Prioritized VCP arms also get `--vcp-temperature 24`, and every invocation runs one agent.
  Rows are validated against the registered blocks, the VCP rules and the shadow report.

## Mid-run readings are not ceilings

This project has repeatedly reported a flat reading as a plateau and had to withdraw it
(`CLAUDE.md`, "The failure mode this project keeps hitting"). Here that means: do not
conclude that an operator removes little, or that knowledge has stopped recovering, from
an early checkpoint or a partial run. Compare against the same agent's trajectory at the
same checkpoint, use full registered horizons, and state results only from registered
points. A knowledge value that has been flat for a long time is not converged.
