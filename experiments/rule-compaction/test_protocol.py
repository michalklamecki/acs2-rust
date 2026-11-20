import copy
import json
from pathlib import Path
import sys
import unittest

from protocol import (COMPACTING_RUN_NAMES, EVALUATION_MAZES, INTERVENTION_POINTS, ROOT, SHADOW_ARMS, SPECIFICATIONS,
                      TUNING_MAZES, TUNING_SEEDS, UNTOUCHED_BY_TUNING, intervention_specifications, load, sha256,
                      sha256_of_text, dump)
from register import INTERVENTION, SHADOW, outputs

sys.path.insert(0, str(ROOT / 'experiments/grid'))

from agent_names import COMPACTING_RUN_NAMES as GRID_COMPACTING_RUN_NAMES
from agent_names import plain_family

SHADOW_PROTOCOL = load(SHADOW)
DRAFT = load(INTERVENTION)
PAIRS = load(ROOT / 'experiments/maze-baselines/pair-protocol.json')


class RegistrationIdentityTests(unittest.TestCase):
    def test_committed_files_are_exactly_what_the_generator_writes(self):
        for path, content in outputs().items():
            with self.subTest(path=path.name):
                self.assertEqual(path.read_text(), content)

    def test_the_pair_protocol_is_pinned_by_content_and_agrees_with_the_earlier_registration(self):
        digest = sha256(ROOT / 'experiments/maze-baselines/pair-protocol.json')
        self.assertEqual(SHADOW_PROTOCOL['pair_protocol_sha256'], digest)
        self.assertEqual(DRAFT['pair_protocol_sha256'], digest)
        self.assertEqual(load(ROOT / 'experiments/value-consistency/plan-draft.json')['pair_protocol_sha256'], digest)
        self.assertEqual(SHADOW_PROTOCOL['protocol'], 'maze-pairs-v1')
        self.assertEqual(PAIRS['id'], 'maze-pairs-v1')

    def test_only_the_shadow_measurement_is_approved_and_the_draft_is_not(self):
        self.assertIs(SHADOW_PROTOCOL['approved'], True)
        self.assertEqual(SHADOW_PROTOCOL['stage'], 'approved_pilot_then_grid')
        self.assertEqual(SHADOW_PROTOCOL['approval']['confirmed_on'], '2026-11-19')
        self.assertEqual(load(ROOT / 'experiments/rule-compaction/approval.json')['plan_sha256'], sha256(SHADOW))
        self.assertIs(DRAFT['approved'], False)
        self.assertEqual(DRAFT['stage'], 'draft_not_approved')
        self.assertIn('must not be submitted', DRAFT['not_approved'])


class ShadowRegistrationTests(unittest.TestCase):
    def test_the_split_matches_the_plan_and_leaves_three_mazes_untouched(self):
        split = SHADOW_PROTOCOL['split']
        self.assertEqual(split['tuning'], dict(mazes=['mazema_c9_pairs', 'woods14_c18_pairs', 'maze4_c5_pairs'],
                                               block='validation', seeds=[42, 43, 44, 45, 46]))
        self.assertEqual(split['evaluation']['block'], 'test')
        self.assertEqual(split['evaluation']['seeds'], list(range(42, 62)))
        self.assertEqual(len(split['evaluation']['mazes']), 6)
        self.assertEqual(split['evaluation']['claims_only_on'], ['maze5_c8_pairs', 'maze6_c10_pairs', 'maze7_c10_pairs'])
        self.assertFalse(set(UNTOUCHED_BY_TUNING) & set(TUNING_MAZES))
        self.assertTrue(set(TUNING_MAZES) | set(UNTOUCHED_BY_TUNING) <= set(EVALUATION_MAZES))
        self.assertEqual(SHADOW_PROTOCOL['seeds'], list(TUNING_SEEDS))

    def test_the_frozen_success_measures_are_the_planned_ones(self):
        helps = SHADOW_PROTOCOL['success_measures']['helps']
        self.assertEqual((helps['minimum_matching_test_reduction'], helps['maximum_knowledge_loss'],
                          helps['maximum_held_out_success_loss']), (0.30, 0.01, 0.01))
        self.assertIn('same agent without compaction', helps['reference'])
        self.assertIn('not used for tuning', helps['mazes'])

    def test_the_arms_are_the_four_families_with_their_uniform_controls(self):
        names = [arm['run_name'] for arm in SHADOW_PROTOCOL['arms']]
        self.assertEqual(names, ['acs2', 'acs2er', 'acs2her', 'acs2vcp_k2', 'acs2vcp_k4',
                                 'acs2vcp_k2_uniform', 'acs2vcp_k4_uniform'])
        by_name = {arm['run_name']: arm for arm in SHADOW_PROTOCOL['arms']}
        self.assertEqual(by_name['acs2vcp_k2_uniform']['control_of'], 'acs2vcp_k2')
        self.assertEqual(by_name['acs2vcp_k4_uniform']['control_of'], 'acs2vcp_k4')
        for name in ('acs2vcp_k2', 'acs2vcp_k4'):
            self.assertEqual(by_name[name]['temperature'], 24)
        for name in ('acs2vcp_k2_uniform', 'acs2vcp_k4_uniform', 'acs2', 'acs2er', 'acs2her'):
            self.assertNotIn('temperature', by_name[name])
        self.assertEqual([arm['heads'] for arm in SHADOW_PROTOCOL['arms']], [1, 1, 1, 2, 4, 2, 4])

    def test_every_arm_has_a_pinned_specification_that_its_agent_can_run(self):
        for arm in SHADOW_PROTOCOL['arms']:
            with self.subTest(arm=arm['run_name']):
                path = ROOT / arm['shadow_specification']
                self.assertEqual(sha256(path), arm['shadow_specification_sha256'])
                specification = load(path)
                labels = [entry['label'] for entry in specification['entries']]
                self.assertEqual(len(labels), len(set(labels)))
                if arm['run_name'] == 'acs2':
                    self.assertEqual(arm['shadow_specification'], 'experiments/rule-compaction/specifications/shadow_without_evidence.json')
                    self.assertEqual(specification['evidence_sample'], 0)
                    self.assertIsNone(specification['audit'])
                    kinds = {operator['kind'] for entry in specification['entries'] for operator in entry['operators']}
                    self.assertFalse(kinds & {'idle_pruning', 'capacity'})
                else:
                    self.assertEqual(arm['shadow_specification'], 'experiments/rule-compaction/specifications/shadow_all_operators.json')
                    self.assertGreater(specification['evidence_sample'], 0)

    def test_the_run_names_the_grid_validators_know_are_the_compacting_forms_of_the_arms(self):
        self.assertEqual(set(COMPACTING_RUN_NAMES), set(GRID_COMPACTING_RUN_NAMES) - {'acs2er_episodic_compacting'})
        for arm in SHADOW_PROTOCOL['arms']:
            self.assertEqual(plain_family(arm['run_name'] + '_compacting'), arm['run_name'])

    def test_configurations_are_the_tuning_mazes_of_the_registered_pair_grid(self):
        registered = {entry['id']: entry for entry in PAIRS['configurations']}
        baseline = {entry['id']: entry for entry in load(ROOT / 'experiments/maze-baselines/grid-plan.json')['configurations']}
        self.assertEqual([entry['id'] for entry in SHADOW_PROTOCOL['configurations']], list(TUNING_MAZES))
        for configuration in SHADOW_PROTOCOL['configurations']:
            identifier = configuration['id']
            self.assertIn(identifier, registered)
            for key in ('task', 'cap', 'budget_steps', 'targets', 'knowledge_transition_goal_pairs', 'protocol'):
                self.assertEqual(configuration[key], baseline[identifier][key])
            self.assertEqual(configuration['cap'], registered[identifier]['cap'])
            self.assertEqual(configuration['evaluation_blocks'], ['validation'])
            self.assertEqual(configuration['targets'][-1], configuration['budget_steps'])
            self.assertEqual(set(configuration['agent_settings']), {arm['run_name'] for arm in SHADOW_ARMS})
            for name, settings in configuration['agent_settings'].items():
                self.assertTrue((ROOT / settings['compaction_shadow']).exists())
                self.assertEqual(settings.get('temperature'), 24 if name in ('acs2vcp_k2', 'acs2vcp_k4') else None)

    def test_the_command_template_uses_options_the_measurement_binary_has(self):
        source = (ROOT / 'acs2-measure/src/main.rs').read_text()
        template = SHADOW_PROTOCOL['command_template']
        for option in ('--protocol', '--evaluation-blocks', '--vcp-temperature', '--compaction-shadow', '--targets', '--seeds'):
            self.assertIn(option, template)
            self.assertIn('"' + option + '"', source)

    def test_the_budget_and_the_mandatory_pilot_are_recorded(self):
        cost = SHADOW_PROTOCOL['cost']
        self.assertEqual(cost['plan_cpu_hour_expectation'], 6)
        self.assertGreater(cost['cpu_hour_ceiling'], 6)
        reservation = cost['pilot_hard_reservation_minutes'] + cost['grid_hard_reservation_minutes']
        self.assertEqual(cost['cpu_hour_ceiling'], -(-reservation // 60))
        self.assertEqual(SHADOW_PROTOCOL['execution']['phase_cpu_hour_limit'], cost['cpu_hour_ceiling'])
        self.assertEqual(SHADOW_PROTOCOL['approval']['cpu_hour_ceiling'], cost['cpu_hour_ceiling'])
        per_seed = sum(sum(limits.values()) for limits in cost['archived_limit_minutes'].values())
        self.assertEqual(cost['archived_limit_minutes_per_seed'], per_seed)
        self.assertEqual(cost['archived_hard_limit_cpu_hours_for_all_seeds_without_shadow'], per_seed * 5 / 60)
        self.assertEqual(set(cost['archived_limit_minutes']), set(TUNING_MAZES))
        pilot = SHADOW_PROTOCOL['cost_pilot']
        self.assertTrue(pilot['mandatory'])
        self.assertEqual(pilot['seed'], 42)
        self.assertEqual(pilot['runs'], 7 * 3)
        self.assertIn('grant', ' '.join(SHADOW_PROTOCOL['before_cluster_use']))
        self.assertIn('Approved by the user', SHADOW_PROTOCOL['before_cluster_use'][0])

    def test_the_comparator_arms_that_the_command_line_cannot_express_are_declared(self):
        source = (ROOT / 'acs2-measure/src/main.rs').read_text()
        self.assertNotIn('"--do-ga"', source)
        self.assertNotIn('"--u-max"', source)
        self.assertIn('A7', SHADOW_PROTOCOL['unexpressed_arms']['comparators'])
        self.assertNotIn('A7', json.dumps(SHADOW_PROTOCOL['arms']))


class InterventionDraftTests(unittest.TestCase):
    def test_the_points_are_ordered_and_inside_every_horizon(self):
        points = [DRAFT['interventions'][name]['interaction'] for name in ('early', 'middle', 'late')]
        self.assertEqual(points, sorted(points))
        self.assertEqual(dict(zip(('early', 'middle', 'late'), points)), INTERVENTION_POINTS)
        for configuration in DRAFT['configurations']:
            self.assertLess(points[-1], configuration['budget_steps'])

    def test_at_least_ten_seeds_and_every_specification_is_pinned(self):
        self.assertGreaterEqual(len(DRAFT['seeds']), 10)
        self.assertEqual(DRAFT['runs_per_agent'], 3 * 3 * len(DRAFT['seeds']))
        for name, point in DRAFT['interventions'].items():
            for variant in ('with_evidence', 'without_evidence'):
                path = ROOT / 'experiments/rule-compaction' / point[variant]
                self.assertEqual(sha256(path), point[variant + '_sha256'])
                specification = load(path)
                self.assertEqual(specification['trigger'], dict(kind='once', at_interaction=point['interaction']))
                self.assertEqual(specification['mode'], 'apply')

    def test_the_single_reduction_specifications_differ_from_the_base_only_in_the_point_and_the_evidence(self):
        base = load(SPECIFICATIONS / 'mature_subsumption_exact_once.json')
        for name, specification in intervention_specifications().items():
            changed = {key for key in base if specification[key] != base[key]}
            expected = {'trigger'} | ({'audit', 'evidence_sample'} if name.endswith('_without_evidence.json') else set())
            self.assertLessEqual(changed, expected)
            if name.endswith('_without_evidence.json'):
                self.assertEqual(specification['audit'], {'policy': 'skip'})
                self.assertEqual(specification['evidence_sample'], 0)

    def test_controls_reuse_only_what_is_equal_and_the_agent_set_is_not_fixed(self):
        self.assertIn('seeds 42-46', DRAFT['controls']['reuse'])
        self.assertEqual(DRAFT['candidate_agents'], list(COMPACTING_RUN_NAMES))
        self.assertIn('Narrowed by the shadow gate', DRAFT['agents_rule'])
        self.assertEqual(DRAFT['agent_settings']['acs2vcp_k2_compacting'], {'temperature': 24})
        self.assertEqual(DRAFT['agent_settings']['acs2vcp_k2_uniform_compacting'], {})


class SpecificationDirectoryTests(unittest.TestCase):
    def test_the_readme_names_every_specification_file(self):
        readme = (ROOT / 'experiments/rule-compaction/README.md').read_text()
        for path in sorted(SPECIFICATIONS.glob('*.json')):
            self.assertIn(path.name, readme, path.name)

    def test_generated_text_hashes_agree_with_files(self):
        for name, specification in intervention_specifications().items():
            self.assertEqual(sha256(SPECIFICATIONS / name), sha256_of_text(dump(specification)))


if __name__ == '__main__':
    unittest.main()
