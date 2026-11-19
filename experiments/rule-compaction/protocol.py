import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPECIFICATIONS = HERE / 'specifications'
PAIR_PROTOCOL = ROOT / 'experiments/maze-baselines/pair-protocol.json'
BASELINE_PLAN = ROOT / 'experiments/maze-baselines/grid-plan.json'
HINDSIGHT_PLAN = ROOT / 'experiments/her-variants/plan.json'
VALUE_CONSISTENCY_PLAN = ROOT / 'experiments/value-consistency/plan-draft.json'

PROTOCOL_ID = 'maze-pairs-v1'
TUNING_MAZES = ('mazema_c9_pairs', 'woods14_c18_pairs', 'maze4_c5_pairs')
EVALUATION_MAZES = ('maze4_c5_pairs', 'maze5_c8_pairs', 'maze6_c10_pairs', 'maze7_c10_pairs',
                    'woods14_c18_pairs', 'mazema_c9_pairs')
UNTOUCHED_BY_TUNING = ('maze5_c8_pairs', 'maze6_c10_pairs', 'maze7_c10_pairs')
TUNING_SEEDS = tuple(range(42, 47))
EVALUATION_SEEDS = tuple(range(42, 62))
INTERVENTION_SEEDS = tuple(range(42, 52))
TEMPERATURE = 24
PLAN_SHADOW_CPU_HOUR_EXPECTATION = 6
INTERVENTION_CPU_HOUR_ESTIMATE = 20

APPROVAL_DATE = '2026-11-19'
PILOT_SEEDS = (42,)
LIMIT_FACTOR = 2
LIMIT_MARGIN_MINUTES = 30
PARTITION = 'lem-cpu-normal'
CPU_MODEL = 'AMD EPYC 9554 64-Core Processor'
MEMORY_MIB = 1024
PILOT_CONCURRENCY = 21
GRID_CONCURRENCY = 35
FIRST_ROW_TIMEOUT_SECONDS = 60
JOB_NAME = 'rule-compaction-shadow'
VCP_DIAGNOSTICS = 'evaluation_disagreement'

SHADOW_WITH_EVIDENCE = 'specifications/shadow_all_operators.json'
SHADOW_WITHOUT_EVIDENCE = 'specifications/shadow_without_evidence.json'
ONCE_BASE = 'mature_subsumption_exact_once.json'
INTERVENTION_POINTS = {'early': 50_000, 'middle': 500_000, 'late': 1_500_000}

SHADOW_ARMS = (
    dict(run_name='acs2', family='acs2', specification=SHADOW_WITHOUT_EVIDENCE, heads=1, control_of=None),
    dict(run_name='acs2er', family='acs2er', specification=SHADOW_WITH_EVIDENCE, heads=1, control_of=None),
    dict(run_name='acs2her', family='acs2her', specification=SHADOW_WITH_EVIDENCE, heads=1, control_of=None),
    dict(run_name='acs2vcp_k2', family='acs2vcp', specification=SHADOW_WITH_EVIDENCE, heads=2,
         temperature=TEMPERATURE, control_of=None),
    dict(run_name='acs2vcp_k4', family='acs2vcp', specification=SHADOW_WITH_EVIDENCE, heads=4,
         temperature=TEMPERATURE, control_of=None),
    dict(run_name='acs2vcp_k2_uniform', family='acs2vcp', specification=SHADOW_WITH_EVIDENCE, heads=2,
         temperature=0, control_of='acs2vcp_k2'),
    dict(run_name='acs2vcp_k4_uniform', family='acs2vcp', specification=SHADOW_WITH_EVIDENCE, heads=4,
         temperature=0, control_of='acs2vcp_k4'),
)

COMPACTING_RUN_NAMES = tuple(arm['run_name'] + '_compacting' for arm in SHADOW_ARMS)

SUCCESS_MEASURES = dict(
    cost='Classifier perception tests in training plus compaction work in the same unit; deterministic and independent of hardware. Wall time, memory and the area under the population-size curve are reported separately.',
    quality='Knowledge and held-out success, as the mean over the registered checkpoints and as the final point, plus the value diagnostics.',
    helps=dict(
        minimum_matching_test_reduction=0.30,
        maximum_knowledge_loss=0.01,
        maximum_held_out_success_loss=0.01,
        reference='the same agent without compaction on the same seeds',
        mazes='mazes not used for tuning',
    ),
    frozen='before any learning run of this line; changing a value after results requires a new registration',
)

UNEXPRESSED_ARMS = dict(
    comparators='A7: GA enabled and a small u_max',
    reason='acs2-measure builds every agent from one fixed thesis preset and has no option for do_ga or u_max, so these arms cannot be run without adding command-line surface.',
    status='not registered; a decision to add the options is a separate scope change',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_of_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def dump(value):
    return json.dumps(value, indent=2) + '\n'


def specification_path(relative):
    return HERE / relative


def intervention_name(point, evidence):
    return f'single_reduction_{point}' + ('' if evidence else '_without_evidence') + '.json'


def intervention_specification(point, evidence):
    base = load(SPECIFICATIONS / ONCE_BASE)
    base['trigger']['at_interaction'] = INTERVENTION_POINTS[point]
    if not evidence:
        base['audit'] = {'policy': 'skip'}
        base['evidence_sample'] = 0
    return base


def intervention_specifications():
    return {intervention_name(point, evidence): intervention_specification(point, evidence)
            for point in INTERVENTION_POINTS for evidence in (True, False)}


def maze_configurations(ids, evaluation_blocks):
    registered = {entry['id'] for entry in load(PAIR_PROTOCOL)['configurations']}
    plan = {entry['id']: entry for entry in load(BASELINE_PLAN)['configurations']}
    keys = ('id', 'task', 'cap', 'pool', 'row_pool', 'encoding', 'row_schema', 'protocol',
            'knowledge_transition_goal_pairs', 'budget_steps', 'targets')
    result = []
    for identifier in ids:
        if identifier not in registered or identifier not in plan:
            raise ValueError('maze is not in the registered pair protocol: ' + identifier)
        result.append(dict({key: plan[identifier][key] for key in keys}, evaluation_blocks=evaluation_blocks))
    return result


def archived_limit_minutes():
    baseline = {entry['id']: entry['limits_minutes'] for entry in load(BASELINE_PLAN)['configurations']}
    hindsight = {entry['id']: entry['limits_minutes'] for entry in load(HINDSIGHT_PLAN)['configurations']}
    ensemble = {entry['id']: entry['limits_minutes'] for entry in load(VALUE_CONSISTENCY_PLAN)['configurations']}
    table = {}
    for identifier in TUNING_MAZES:
        limits = {name: baseline[identifier][name] for name in ('acs2', 'acs2er')}
        limits['acs2her'] = hindsight[identifier]['her_full']
        limits.update({name: ensemble[identifier][name] for name in
                       ('acs2vcp_k2', 'acs2vcp_k4', 'acs2vcp_k2_uniform', 'acs2vcp_k4_uniform')})
        table[identifier] = {arm['run_name']: limits[arm['run_name']] for arm in SHADOW_ARMS}
    return table


def shadow_limit_minutes(archived):
    return max(LIMIT_FACTOR * archived, archived + LIMIT_MARGIN_MINUTES)


def registered_limit_minutes():
    return {identifier: {name: shadow_limit_minutes(minutes) for name, minutes in arms.items()}
            for identifier, arms in archived_limit_minutes().items()}


def array_limit_minutes(limits):
    return {identifier: max(arms.values()) for identifier, arms in limits.items()}


def hard_reservation_minutes(limits):
    array = array_limit_minutes(limits)
    arms = len(SHADOW_ARMS)
    pilot = len(PILOT_SEEDS) * arms * len(array) * max(array.values())
    grid = len(TUNING_SEEDS) * arms * sum(array.values())
    return dict(array=array, pilot=pilot, grid=grid)


def cpu_hour_ceiling(limits):
    reservation = hard_reservation_minutes(limits)
    return -(-(reservation['pilot'] + reservation['grid']) // 60)


def shadow_cost():
    limits = archived_limit_minutes()
    registered = registered_limit_minutes()
    reservation = hard_reservation_minutes(registered)
    per_seed_minutes = sum(sum(arm.values()) for arm in limits.values())
    return dict(
        plan_cpu_hour_expectation=PLAN_SHADOW_CPU_HOUR_EXPECTATION,
        cpu_hour_ceiling=cpu_hour_ceiling(registered),
        ceiling_origin='sum of the registered hard limits of the pilot and the grid, rounded up to whole CPU-hours; the user approved a ceiling above the plan expectation on ' + APPROVAL_DATE,
        limit_rule=f'registered limit = max({LIMIT_FACTOR} x archived limit, archived limit + {LIMIT_MARGIN_MINUTES} minutes); shadow overhead is unknown, a timeout wastes a run and the reservation is cheap against the grant',
        archived_limit_minutes=limits,
        array_limit_minutes=reservation['array'],
        pilot_hard_reservation_minutes=reservation['pilot'],
        grid_hard_reservation_minutes=reservation['grid'],
        pilot_hard_reservation_cpu_seconds=reservation['pilot'] * 60,
        hard_reservation_arithmetic='An array element is reserved at the largest limit of its array. Pilot: one array of 21 runs at the largest limit of any maze. Grid: one array of 35 runs per maze at that maze limit. The grid manifest carries the whole pilot reservation as its pilot allocation, so pilot plus grid never exceed the ceiling.',
        archived_limit_minutes_per_seed=per_seed_minutes,
        archived_hard_limit_cpu_hours_for_all_seeds_without_shadow=per_seed_minutes * len(TUNING_SEEDS) / 60,
        archived_hard_limit_cpu_hours_for_the_pilot_without_shadow=per_seed_minutes / 60,
        limit_source='hard limits of the archived registrations of the same agents on the same mazes: maze-baselines/grid-plan.json, her-variants/plan.json (her_full), value-consistency/plan-draft.json',
        accounting_note='experiments/grid charges every array element at its hard limit against the phase limit, so the sum of the limits, not the expected usage, must fit the approved limit. The archived limits exclude shadow overhead; the registered limits carry the margin for it.',
        decision_after_pilot='The limits and the ceiling are registered and approved. The pilot reports the measured cost beside them; changing a limit, a seed or an arm afterwards needs a new registration and a new approval, and nothing is reduced silently.',
    )


def shadow_execution(cost):
    return dict(
        partition=PARTITION,
        cpu_model=CPU_MODEL,
        memory_mib=MEMORY_MIB,
        memory_note='512 MiB in the archived registrations; doubled because the shadow keeps copies of every head population and an evidence sample',
        job_name=JOB_NAME,
        first_row_timeout_seconds=FIRST_ROW_TIMEOUT_SECONDS,
        pilot=dict(stage='authorized_cost_pilot', seeds=list(PILOT_SEEDS), concurrency=PILOT_CONCURRENCY,
                   runs=len(SHADOW_ARMS) * len(TUNING_MAZES), pilot_allocation_cpu_seconds=0),
        grid=dict(stage='approved_grid', seeds=list(TUNING_SEEDS), concurrency=GRID_CONCURRENCY,
                  runs=len(SHADOW_ARMS) * len(TUNING_MAZES) * len(TUNING_SEEDS),
                  runs_per_array=len(SHADOW_ARMS) * len(TUNING_SEEDS),
                  pilot_allocation_cpu_seconds=cost['pilot_hard_reservation_cpu_seconds'],
                  seed_42='run again in the grid so the grid is one self-contained set; its learning fields must equal the pilot rows'),
        phase_cpu_hour_limit=cost['cpu_hour_ceiling'],
    )


def shadow_protocol():
    arms = []
    settings = {}
    for arm in SHADOW_ARMS:
        entry = dict(run_name=arm['run_name'], family=arm['family'], heads=arm['heads'],
                     control_of=arm['control_of'], shadow_specification=arm['specification'],
                     shadow_specification_sha256=sha256(specification_path(arm['specification'])))
        setting = dict(compaction_shadow=arm['specification'],
                       compaction_shadow_sha256=entry['shadow_specification_sha256'])
        if arm.get('temperature'):
            entry['temperature'] = arm['temperature']
            setting['temperature'] = arm['temperature']
        arms.append(entry)
        settings[arm['run_name']] = setting
    configurations = maze_configurations(TUNING_MAZES, ['validation'])
    limits = registered_limit_minutes()
    for configuration in configurations:
        configuration['agent_settings'] = settings
        configuration['limits_minutes'] = limits[configuration['id']]
        configuration['vcp_diagnostics'] = VCP_DIAGNOSTICS
    cost = shadow_cost()
    return dict(
        schema=1,
        stage='approved_pilot_then_grid',
        approved=True,
        approval=dict(
            record='approval.json',
            confirmed_on=APPROVAL_DATE,
            scope='shadow measurement: the seed-42 cost pilot and then the full grid of seeds 42-46, at the registered limits',
            cpu_hour_ceiling=cost['cpu_hour_ceiling'],
            plan_cpu_hour_expectation=PLAN_SHADOW_CPU_HOUR_EXPECTATION,
            note='The user allowed a budget above the plan expectation; the record carries the hash of this file.',
        ),
        experiment='rule-compaction',
        measurement='shadow_compaction',
        purpose='Measure what each operator would remove from the populations of unchanged learning runs, and what that would do to coverage, values and greedy actions, at every registered checkpoint. Nothing is applied.',
        protocol=PROTOCOL_ID,
        pair_protocol_sha256=sha256(PAIR_PROTOCOL),
        split=dict(
            tuning=dict(mazes=list(TUNING_MAZES), block='validation', seeds=list(TUNING_SEEDS)),
            evaluation=dict(mazes=list(EVALUATION_MAZES), block='test', seeds=list(EVALUATION_SEEDS),
                            claims_only_on=list(UNTOUCHED_BY_TUNING)),
            this_measurement='tuning',
        ),
        success_measures=SUCCESS_MEASURES,
        seeds=list(TUNING_SEEDS),
        arms=arms,
        unexpressed_arms=UNEXPRESSED_ARMS,
        command_template='acs2-measure --task TASK --cap CAP --pool full --encoding coordinates --protocol maze-pairs-v1 --evaluation-blocks validation --agents RUN_NAME [--vcp-temperature 24] --compaction-shadow SPECIFICATION --seeds SEED --targets TARGETS --out OUT',
        learning_unchanged='Shadow runs on copies with strided evidence and draws no random numbers, so every learning field of a row equals the same run without the shadow report.',
        configurations=configurations,
        cost=cost,
        execution=shadow_execution(cost),
        cost_pilot=dict(
            mandatory=True,
            seed=42,
            runs=len(SHADOW_ARMS) * len(TUNING_MAZES),
            rule='Run every arm on every tuning maze at the registered budgets with seed 42 before the grid. Record allocation CPU, process CPU, wall time and the population-size area per run, and report them beside the registered limits. The user approved the grid together with the pilot, so the grid follows when the pilot completes cleanly.',
            identity='Where verified archived rows of the same agent exist, the seed-42 learning fields must reproduce them bit for bit.',
        ),
        before_cluster_use=[
            'Approved by the user on ' + APPROVAL_DATE + ' (approval.json); the approval record must carry the hash of this file.',
            'Check the grant: sshare -U -u "$WCSS_USER" -A "$WCSS_ACCOUNT" -o RawUsage -n against WCSS_GRANT_CPU_HOURS, plus the queue commitment.',
            'Done: experiments/rule-compaction/execution.py derives the pilot and grid manifests, grid_authorization.py dispatches to it, and collect.py passes agent_settings.compaction_shadow to --compaction-shadow after verifying its SHA-256.',
            'Build the musl binary once in a dedicated clone at a pinned clean commit; never rebuild it on the cluster.',
            'Submit the seed-42 pilot, wait for all 21 runs, check them and report the measured cost.',
            'Submit the grid only after the pilot completed without a STOP or a timeout; a limit that was hit means a new registration.',
        ],
    )


def intervention_draft():
    names = {point: {evidence: 'specifications/' + intervention_name(point, evidence) for evidence in (True, False)}
             for point in INTERVENTION_POINTS}
    specifications = {}
    for point, variants in names.items():
        specifications[point] = dict(
            interaction=INTERVENTION_POINTS[point],
            with_evidence=variants[True],
            with_evidence_sha256=sha256_of_text(dump(intervention_specification(point, True))),
            without_evidence=variants[False],
            without_evidence_sha256=sha256_of_text(dump(intervention_specification(point, False))),
        )
    configurations = maze_configurations(TUNING_MAZES, ['validation'])
    return dict(
        schema=1,
        stage='draft_not_approved',
        approved=False,
        experiment='rule-compaction',
        measurement='single_reduction',
        not_approved='This is a draft. It must not be submitted, pilot included. The operators, the agents and the intervention points are placeholders until the shadow measurement has been analysed and its gate has chosen at most three operators.',
        purpose='Apply one reduction early, in the middle or late in learning, then let learning continue unchanged. Measure regrowth of the population, the share of removed rules that return, the recovery of knowledge and success, and the matching tests saved.',
        protocol=PROTOCOL_ID,
        pair_protocol_sha256=sha256(PAIR_PROTOCOL),
        split=dict(tuning=dict(mazes=list(TUNING_MAZES), block='validation')),
        seeds=list(INTERVENTION_SEEDS),
        seeds_rule='At least ten seeds. After a reduction the trajectories diverge, because action selection draws randomness in proportion to the number of matching rules, so the comparison is statistical.',
        operator='mature_subsumption exact (guarded, tolerance 0) as the base specification mature_subsumption_exact_once.json; the operator set is revisited after the shadow gate',
        interventions=specifications,
        candidate_agents=list(COMPACTING_RUN_NAMES),
        agents_rule='Narrowed by the shadow gate; ACS2 uses the without-evidence specifications.',
        agent_settings={name: ({'temperature': TEMPERATURE} if name.startswith('acs2vcp') and 'uniform' not in name else {})
                        for name in COMPACTING_RUN_NAMES},
        configurations=configurations,
        runs_per_agent=len(TUNING_MAZES) * len(INTERVENTION_POINTS) * len(INTERVENTION_SEEDS),
        controls=dict(
            reference='the same agent without compaction on the same seeds and mazes',
            reuse='Shadow runs leave learning unchanged, so seeds 42-46 of the shadow measurement are valid controls once their identity is verified; seeds 47-51 need new plain runs.',
        ),
        measures=['population size after the reduction and its regrowth',
                  'share of removed rules that are recreated (removal ledger restoration)',
                  'knowledge and held-out success against the control at every checkpoint',
                  'matching tests saved, net of compaction work'],
        gate='A lasting net saving without a delay in learning, and a map of the moments at which a reduction hurts.',
        budget=dict(plan_estimate_cpu_hours=INTERVENTION_CPU_HOUR_ESTIMATE,
                    status='estimate from the plan, not a forecast; a forecast comes from the pilot of the approved agent set'),
        before_cluster_use=['Analyse the shadow measurement and apply its gate.',
                            'Replace the placeholders, register the final agent set, operators and points, and have the user approve the registration.',
                            'Run the seed-42 cost pilot first; check the grant before every submission.'],
    )
