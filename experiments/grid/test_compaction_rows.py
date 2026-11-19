import copy
import json
from pathlib import Path
import unittest

from agent_names import COMPACTING_RUN_NAMES, canonical_run_name, is_compacting, plain_family
from collect import validate_row
from compaction_schema import head_count, validate_compaction

ROOT = Path(__file__).resolve().parents[2]
ROWS = json.loads(Path(__file__).with_name('compacting_rows.json').read_text())
PROTOCOL = json.loads((ROOT / 'experiments/maze-baselines/pair-protocol.json').read_text())['configurations'][0]
GRID = json.loads((ROOT / 'experiments/maze-baselines/grid-plan.json').read_text())['configurations'][0]


def config(entry):
    return dict(id=GRID['id'], task='maze4', cap=5, pool='full', row_pool='full', encoding='coordinates',
                row_schema=5, protocol='maze-pairs-v1', registration=PROTOCOL,
                knowledge_transition_goal_pairs=GRID['knowledge_transition_goal_pairs'],
                targets=[entry['target']], agent_settings={entry['run']: entry['settings']})


def validate(entry, row=None, **changes):
    row = copy.deepcopy(entry['row'] if row is None else row)
    row.update(changes)
    validate_row(row, config(entry), entry['run'], 42, 'a' * 40, entry['target'])


def entry_of(run):
    return next(entry for entry in ROWS if entry['run'] == run)


class CompactingNamesTests(unittest.TestCase):
    def test_exactly_the_eight_run_names_map_to_their_plain_family(self):
        self.assertEqual(set(COMPACTING_RUN_NAMES), {
            'acs2_compacting', 'acs2er_compacting', 'acs2er_episodic_compacting', 'acs2her_compacting',
            'acs2vcp_k2_compacting', 'acs2vcp_k4_compacting',
            'acs2vcp_k2_uniform_compacting', 'acs2vcp_k4_uniform_compacting'})
        for name in COMPACTING_RUN_NAMES:
            self.assertTrue(is_compacting(name))
            self.assertEqual(plain_family(name) + '_compacting', name)
            self.assertFalse(is_compacting(plain_family(name)))
        for name in ('acs2', 'acs2vcp_k2', 'acs2her_values', 'acs2vcp_k3_compacting', 'acs2er_one_update_compacting'):
            self.assertFalse(is_compacting(name))
            self.assertEqual(plain_family(name), name)

    def test_archived_spellings_still_resolve_to_their_current_plain_names(self):
        self.assertEqual(plain_family('her_full'), 'acs2her')
        self.assertEqual(plain_family('er_episodic'), 'acs2er_episodic')
        self.assertEqual(canonical_run_name('acs2vcp_k2_compacting'), 'acs2vcp_k2_compacting')

    def test_head_counts_follow_the_family(self):
        self.assertEqual([head_count(plain_family(name)) for name in sorted(COMPACTING_RUN_NAMES)],
                         [1, 1, 1, 1, 2, 2, 4, 4])


class CompactingRowTests(unittest.TestCase):
    def test_real_rows_of_every_family_validate_like_their_plain_family_plus_the_report(self):
        for entry in ROWS:
            with self.subTest(run=entry['run']):
                validate(entry)

    def test_the_fixture_covers_every_family_and_both_ensemble_clocks_of_cost(self):
        self.assertEqual({entry['run'] for entry in ROWS} | {'acs2vcp_k4_compacting', 'acs2vcp_k2_uniform_compacting'},
                         set(COMPACTING_RUN_NAMES))

    def test_the_report_belongs_exactly_to_the_compacting_names(self):
        for entry in ROWS:
            with self.subTest(run=entry['run']):
                row = copy.deepcopy(entry['row'])
                row.pop('compaction')
                with self.assertRaisesRegex(ValueError, 'belongs exactly'):
                    validate(entry, row)
        plain = copy.deepcopy(entry_of('acs2er_compacting'))
        plain['run'] = 'acs2er'
        with self.assertRaisesRegex(ValueError, 'belongs exactly'):
            validate_compaction(plain['row'], config(plain), 'acs2er')

    def test_a_plain_row_validates_without_any_compaction_key(self):
        row = {'agent_parameters': {}, 'agent': 'acs2er'}
        validate_compaction(row, {}, 'acs2er')
        validate_compaction(row, {}, 'acs2vcp_k2')

    def test_a_row_must_name_its_own_family_and_update_counts(self):
        entry = entry_of('acs2er_compacting')
        with self.assertRaisesRegex(ValueError, 'agent'):
            validate(entry, agent='acs2er')
        with self.assertRaisesRegex(ValueError, 'update counts'):
            validate(entry, replay_updates=entry['row']['replay_updates'] + 3)
        acs2 = entry_of('acs2_compacting')
        with self.assertRaisesRegex(ValueError, 'update counts'):
            validate(acs2, online_updates=acs2['row']['online_updates'] + 1)

    def test_the_specification_echo_and_stream_are_required(self):
        entry = entry_of('acs2er_compacting')
        row = copy.deepcopy(entry['row'])
        row['agent_parameters']['compaction'].pop('trigger')
        with self.assertRaisesRegex(ValueError, 'complete compaction specification'):
            validate(entry, row)
        row = copy.deepcopy(entry['row'])
        row['agent_parameters']['compaction_stream'] = 12
        with self.assertRaisesRegex(ValueError, 'own stream'):
            validate(entry, row)
        row = copy.deepcopy(entry['row'])
        row['agent_parameters']['compaction']['mode'] = 'sometimes'
        with self.assertRaisesRegex(ValueError, 'mode'):
            validate(entry, row)

    def test_a_registered_specification_must_match_the_echo(self):
        entry = entry_of('acs2her_compacting')
        registered = copy.deepcopy(entry['row']['agent_parameters']['compaction'])
        configured = config(entry)
        configured['agent_settings'][entry['run']] = dict(compaction=registered)
        validate_row(copy.deepcopy(entry['row']), configured, entry['run'], 42, 'a' * 40, entry['target'])
        registered['evidence_sample'] += 1
        with self.assertRaisesRegex(ValueError, 'registered one'):
            validate_row(copy.deepcopy(entry['row']), configured, entry['run'], 42, 'a' * 40, entry['target'])

    def test_totals_costs_and_records_are_cross_checked(self):
        entry = entry_of('acs2vcp_k4_uniform_compacting')
        mutations = {
            'compaction totals disagree': lambda c: c.update(accepted=c['accepted'] + 1),
            'per-head compaction counts': lambda c: c['heads'][0].update(reductions=c['heads'][0]['reductions'] + 1),
            'per-head training cost': lambda c: c['training_cost'].update(match_formations=c['training_cost']['match_formations'] + 1),
            'population size area': lambda c: c['population_size_area'].update(interactions=c['population_size_area']['interactions'] + 1),
            'work accounting': lambda c: c['work'].update(uncharged=c['work']['uncharged'] + 1),
            'record accounting': lambda c: c.update(records_omitted=c['records_omitted'] + 1),
            'every head once': lambda c: c['heads'].pop(),
            'report keys': lambda c: c.pop('vcp'),
            'record sizes': lambda c: c['records'][0].update(size_after=c['records'][0]['size_before'] + 1),
            'record head or mode': lambda c: c['records'][0].update(mode='shadow'),
            'record limit': lambda c: c['records'].extend(copy.deepcopy(c['records'])),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message):
                row = copy.deepcopy(entry['row'])
                mutate(row['compaction'])
                if message == 'record limit':
                    row['agent_parameters']['compaction']['max_records_per_row'] = 5
                    row['compaction']['records_new'] = len(row['compaction']['records'])
                    row['compaction']['records_omitted'] = 0
                with self.assertRaisesRegex(ValueError, message):
                    validate(entry, row)

    def test_off_and_shadow_modes_cannot_report_applied_reductions(self):
        entry = entry_of('acs2er_compacting')
        row = copy.deepcopy(entry['row'])
        row['agent_parameters']['compaction']['mode'] = 'shadow'
        row['compaction']['applied'] = 1
        row['compaction']['accepted'] = max(1, row['compaction']['accepted'])
        row['compaction']['reductions'] = row['compaction']['accepted'] + row['compaction']['rejected']
        with self.assertRaisesRegex(ValueError, 'mode disagrees'):
            validate(entry, row)

    def test_only_the_ensemble_reports_the_vcp_section(self):
        entry = entry_of('acs2er_compacting')
        row = copy.deepcopy(entry['row'])
        row['compaction']['vcp'] = {}
        with self.assertRaisesRegex(ValueError, 'report keys'):
            validate(entry, row)

    def test_a_compacting_ensemble_row_keeps_the_ensemble_identities(self):
        entry = entry_of('acs2vcp_k2_compacting')
        with self.assertRaisesRegex(ValueError, 'derived matching cost'):
            validate(entry, match_formations_train=entry['row']['match_formations_train'] + 1)
        with self.assertRaisesRegex(ValueError, 'update counts'):
            validate(entry, replay_updates=entry['row']['replay_updates'] + 3)
        row = copy.deepcopy(entry['row'])
        row['replay_diagnostics']['priority']['value_queries'] += 1
        with self.assertRaisesRegex(ValueError, 'VCP priority queries'):
            validate(entry, row)
        configured = config(entry)
        configured['agent_settings'][entry['run']] = dict(temperature=12)
        with self.assertRaisesRegex(ValueError, 'VCP'):
            validate_row(copy.deepcopy(entry['row']), configured, entry['run'], 42, 'a' * 40, entry['target'])


class CompactionShadowTests(unittest.TestCase):
    def test_the_fixture_holds_a_shadow_for_a_single_head_and_an_ensemble(self):
        for run in ('acs2_compacting', 'acs2vcp_k2_compacting'):
            self.assertIn('compaction_shadow', entry_of(run)['row'])

    def test_a_plain_row_carries_a_shadow_report_when_present(self):
        entry = entry_of('acs2_compacting')
        row = copy.deepcopy(entry['row'])
        row['compaction_shadow']['interactions'] += 1
        with self.assertRaisesRegex(ValueError, 'every training interaction'):
            validate(entry, row)
        row = copy.deepcopy(entry['row'])
        row['compaction_shadow']['entries'][0]['heads'][0]['applied'] = True
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            validate(entry, row)
        row = copy.deepcopy(entry['row'])
        row['compaction_shadow']['entries'][0]['heads'].append(copy.deepcopy(row['compaction_shadow']['entries'][0]['heads'][0]))
        with self.assertRaisesRegex(ValueError, 'one record per head'):
            validate(entry, row)
        row = copy.deepcopy(entry['row'])
        row['compaction_shadow']['entries'][0]['label'] = 'renamed'
        with self.assertRaisesRegex(ValueError, 'specification'):
            validate(entry, row)

    def test_shadow_on_a_plain_family_row_is_accepted_and_checked(self):
        for run, family in (('acs2_compacting', 'acs2'), ('acs2vcp_k2_compacting', 'acs2vcp_k2')):
            row = copy.deepcopy(entry_of(run)['row'])
            for key in ('compaction',):
                row.pop(key)
            row['agent'] = family
            for key in ('compaction', 'compaction_stream'):
                row['agent_parameters'].pop(key)
            validate_compaction(row, {}, family)
            row['compaction_shadow']['entries'][0]['heads'].pop()
            with self.assertRaisesRegex(ValueError, 'one record per head'):
                validate_compaction(row, {}, family)

    def test_a_registered_shadow_must_be_present(self):
        entry = entry_of('acs2er_compacting')
        configured = config(entry)
        configured['agent_settings'][entry['run']] = dict(compaction_shadow='specifications/shadow_all_operators.json')
        with self.assertRaisesRegex(ValueError, 'registered compaction shadow'):
            validate_row(copy.deepcopy(entry['row']), configured, entry['run'], 42, 'a' * 40, entry['target'])


class ExistingNamesTests(unittest.TestCase):
    def test_unregistered_ensemble_names_are_still_refused(self):
        from row_schema import validate_replay
        for name in ('acs2vcp_k3', 'acs2vcp_k3_compacting'):
            with self.assertRaisesRegex(ValueError, 'unregistered VCP arm'):
                validate_replay({'agent_parameters': {}}, {}, name)

    def test_a_shadow_field_on_a_plain_row_without_registration_is_validated_not_required(self):
        validate_compaction({'agent_parameters': {}}, {'agent_settings': {'acs2': {}}}, 'acs2')


if __name__ == '__main__':
    unittest.main()
