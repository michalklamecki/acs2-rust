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
SHADOW_CPU_HOUR_CEILING = 6
INTERVENTION_CPU_HOUR_ESTIMATE = 20

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


def shadow_cost():
    limits = archived_limit_minutes()
    per_seed_minutes = sum(sum(arm.values()) for arm in limits.values())
    return dict(
        expected_cpu_hour_ceiling=SHADOW_CPU_HOUR_CEILING,
        ceiling_origin='stage budget of the plan, expected allocation CPU including the pilot',
        archived_limit_minutes=limits,
        archived_limit_minutes_per_seed=per_seed_minutes,
        archived_hard_limit_cpu_hours_for_all_seeds_without_shadow=per_seed_minutes * len(TUNING_SEEDS) / 60,
        archived_hard_limit_cpu_hours_for_the_pilot_without_shadow=per_seed_minutes / 60,
        limit_source='hard limits of the archived registrations of the same agents on the same mazes: maze-baselines/grid-plan.json, her-variants/plan.json (her_full), value-consistency/plan-draft.json',
        accounting_note='experiments/grid charges every array element at its hard limit against the phase limit, so the sum of the limits, not the expected usage, must fit the approved limit. Shadow overhead is not in these limits.',
        decision_after_pilot='Set limits from the seed-42 pilot (shadow included). If the hard limits for all seeds exceed the approved phase limit, the user chooses between a higher ceiling, fewer seeds and fewer arms; nothing is reduced silently.',
    )


def shadow_protocol():
    arms = []
    settings = {}
    for arm in SHADOW_ARMS:
        entry = dict(run_name=arm['run_name'], family=arm['family'], heads=arm['heads'],
                     control_of=arm['control_of'], shadow_specification=arm['specification'],
                     shadow_specification_sha256=sha256(specification_path(arm['specification'])))
        setting = dict(compaction_shadow=arm['specification'])
        if arm.get('temperature'):
            entry['temperature'] = arm['temperature']
            setting['temperature'] = arm['temperature']
        arms.append(entry)
        settings[arm['run_name']] = setting
    configurations = maze_configurations(TUNING_MAZES, ['validation'])
    for configuration in configurations:
        configuration['agent_settings'] = settings
    return dict(
        schema=1,
        stage='registered_awaiting_user_approval',
        approved=False,
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
        cost=shadow_cost(),
        cost_pilot=dict(
            mandatory=True,
            seed=42,
            runs=len(SHADOW_ARMS) * len(TUNING_MAZES),
            rule='Run every arm on every tuning maze at the registered budgets with seed 42 before the grid. Record allocation CPU, process CPU, wall time and the population-size area per run. Forecast the grid as the pilot cost times the number of seeds and report it with the sum of hard limits; the grid is submitted only after the user approves that forecast.',
            identity='Where verified archived rows of the same agent exist, the seed-42 learning fields must reproduce them bit for bit.',
        ),
        before_cluster_use=[
            'The user approves this registration and the pilot, in writing.',
            'Check the grant: sshare -U -u "$WCSS_USER" -A "$WCSS_ACCOUNT" -o RawUsage -n against WCSS_GRANT_CPU_HOURS, plus the queue commitment.',
            'Wire the shared grid path for this experiment: an execution.py with --derive-manifest registered in experiments/grid/grid_authorization.py, and the pass-through of agent_settings.compaction_shadow to the --compaction-shadow flag in experiments/grid/collect.py.',
            'Build the musl binary once in a dedicated clone at a pinned clean commit; never rebuild it on the cluster.',
            'Submit the seed-42 pilot, then stop and report.',
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
