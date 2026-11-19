import json
from pathlib import Path

ARCHIVED_RUN_NAMES = json.loads(Path(__file__).with_suffix('.json').read_text())
PLAIN_FAMILIES = {name + '_compacting': name for name in (
    'acs2', 'acs2er', 'acs2er_episodic', 'acs2her',
    'acs2vcp_k2', 'acs2vcp_k4', 'acs2vcp_k2_uniform', 'acs2vcp_k4_uniform')}
COMPACTING_RUN_NAMES = tuple(PLAIN_FAMILIES)


def canonical_run_name(name):
    return ARCHIVED_RUN_NAMES.get(name, name)


def plain_family(name):
    return PLAIN_FAMILIES.get(canonical_run_name(name), canonical_run_name(name))


def is_compacting(name):
    return canonical_run_name(name) in PLAIN_FAMILIES


def canonical_row(row):
    result = dict(row)
    if 'agent' in result:
        result['agent'] = canonical_run_name(result['agent'])
    return result


def agent_settings(config, agent):
    matches = [settings for name, settings in config.get('agent_settings', {}).items()
               if canonical_run_name(name) == canonical_run_name(agent)]
    if len(matches) > 1:
        raise ValueError('duplicate settings for an agent under archived and current names')
    return matches[0] if matches else {}
