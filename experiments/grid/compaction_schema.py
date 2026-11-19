from agent_names import plain_family

SPECIFICATION_KEYS = {'acceptance_limits', 'audit', 'census', 'clock', 'disagreement_probe', 'evidence_sample',
                      'feedback', 'max_records_per_row', 'mode', 'operators', 'phase_modulation', 'strength',
                      'strength_bounds', 'theta_r', 'trigger'}
REPORT_KEYS = {'accepted', 'applied', 'heads', 'population_size_area', 'records', 'records_new', 'records_omitted',
               'records_scope', 'reductions', 'rejected', 'removed_applied', 'removed_candidates',
               'synchronous_policy', 'training_cost', 'training_cost_definition', 'work'}
SHADOW_KEYS = {'entries', 'evidence_definition', 'evidence_size', 'interactions', 'population_size_area',
               'specification'}
COMPACTION_STREAM = 11


def head_count(family):
    if not family.startswith('acs2vcp'):
        return 1
    return 2 if '_k2' in family else 4


def count(value):
    return type(value) is int and value >= 0


def validate_area(area, steps):
    if not count(area.get('interactions')) or not count(area.get('area')) or area['interactions'] != steps:
        raise ValueError('population size area must cover every training interaction')
    expected_mean = area['area'] / area['interactions'] if area['interactions'] else None
    if area['mean_size'] != expected_mean:
        raise ValueError('population size area mean differs')


def validate_specification(parameters, config, agent):
    if not isinstance(parameters, dict) or set(parameters) != SPECIFICATION_KEYS:
        raise ValueError('compacting rows echo the complete compaction specification')
    if parameters['mode'] not in ('off', 'shadow', 'apply'):
        raise ValueError('unknown compaction mode')
    registered = config.get('agent_settings', {}).get(agent, {}).get('compaction')
    if registered is not None and parameters != registered:
        raise ValueError('compaction specification differs from the registered one')
    return parameters


def validate_record(record, mode, heads):
    if not 0 <= record['head'] < heads or record['mode'] != mode:
        raise ValueError('reduction record head or mode differs')
    before, candidate, after = record['size_before'], record['size_candidate'], record['size_after']
    if not 0 <= candidate <= before or after != (candidate if record['applied'] else before):
        raise ValueError('reduction record sizes disagree with its decision')
    if mode == 'shadow' and record['applied']:
        raise ValueError('a shadow reduction cannot be applied')
    if record['applied'] and not record['accepted']:
        raise ValueError('an applied reduction must be accepted')


def validate_compaction(row, config, agent):
    family = plain_family(agent)
    compacting = family != agent
    if compacting != ('compaction' in row):
        raise ValueError('the compaction report belongs exactly to the compacting run names')
    if 'compaction_shadow' in row:
        validate_shadow(row, family)
    elif config.get('agent_settings', {}).get(agent, {}).get('compaction_shadow') is not None:
        raise ValueError('row lacks the registered compaction shadow measurement')
    if not compacting:
        return
    parameters = validate_specification(row['agent_parameters'].get('compaction'), config, agent)
    if row['agent_parameters'].get('compaction_stream') != COMPACTION_STREAM:
        raise ValueError('compaction randomness must use its own stream')
    report, mode, heads = row['compaction'], parameters['mode'], head_count(family)
    expected = REPORT_KEYS | ({'vcp'} if family.startswith('acs2vcp') else set())
    if set(report) != expected:
        raise ValueError('compaction report keys differ')
    totals = [report[key] for key in ('reductions', 'accepted', 'applied', 'rejected', 'removed_applied', 'removed_candidates')]
    if not all(map(count, totals)):
        raise ValueError('compaction totals must be non-negative integers')
    if (report['reductions'] != report['accepted'] + report['rejected'] or report['applied'] > report['accepted']
            or report['removed_applied'] > report['removed_candidates']):
        raise ValueError('compaction totals disagree')
    if mode == 'off' and report['reductions'] or mode == 'shadow' and (report['applied'] or report['removed_applied']):
        raise ValueError('compaction mode disagrees with its reductions')
    details = report['heads']
    if [head['head'] for head in details] != list(range(heads)):
        raise ValueError('compaction report must list every head once')
    if (sum(head['reductions'] for head in details) != report['reductions']
            or sum(head['applied'] for head in details) != report['applied']):
        raise ValueError('per-head compaction counts disagree with the totals')
    cost = report['training_cost']
    if any(sum(head['training_cost'][key] for head in details) != cost[key] for key in ('match_formations', 'classifier_tests')):
        raise ValueError('per-head training cost disagrees with the total')
    if cost['match_formations'] != row['match_formations_train']:
        raise ValueError('derived matching cost differs from the recorded match formations')
    validate_area(report['population_size_area'], row['actual_steps'])
    work = report['work']
    spent = work['operator'] + work['audit'] + work['check'] + work['census']
    if not all(map(count, (work[key] for key in ('operator', 'audit', 'check', 'census', 'charged', 'uncharged')))):
        raise ValueError('compaction work must be non-negative integers')
    if work['charged'] > spent or work['uncharged'] != spent - work['charged']:
        raise ValueError('compaction work accounting differs')
    records = report['records']
    if report['records_scope'] != 'reductions_since_previous_row' or report['records_new'] - report['records_omitted'] != len(records):
        raise ValueError('compaction record accounting differs')
    if len(records) > parameters['max_records_per_row']:
        raise ValueError('compaction row exceeds its record limit')
    for record in records:
        validate_record(record, mode, heads)


def validate_shadow(row, family):
    shadow, heads = row['compaction_shadow'], head_count(family)
    if set(shadow) != SHADOW_KEYS:
        raise ValueError('compaction shadow keys differ')
    if shadow['interactions'] != row['actual_steps']:
        raise ValueError('compaction shadow must observe every training interaction')
    validate_area(shadow['population_size_area'], row['actual_steps'])
    labels = [entry['label'] for entry in shadow['entries']]
    if not labels or len(set(labels)) != len(labels) or labels != [entry['label'] for entry in shadow['specification']['entries']]:
        raise ValueError('shadow entries differ from their specification')
    for entry in shadow['entries']:
        if set(entry) != {'label', 'heads'} or [head['head'] for head in entry['heads']] != list(range(heads)):
            raise ValueError('shadow entry must hold one record per head')
        for record in entry['heads']:
            if record['mode'] != 'shadow' or record['applied'] or record['size_after'] != record['size_before'] or record['size_candidate'] > record['size_before']:
                raise ValueError('a shadow record must leave the population unchanged')
