import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from execution import (APPROVAL, GRID_STAGE, PILOT_STAGE, REGISTRATION, approved_registration,
                       authorization_manifest, grid_manifest, pilot_manifest, printed_manifest)
from protocol import ROOT, SHADOW_ARMS, TUNING_MAZES, TUNING_SEEDS, load, sha256

sys.path.append(str(ROOT / 'experiments/grid'))

from grid_authorization import derive_manifest
from grid_protocol import atomic_json
import collect
from submit import prepare, require_authorization, submit

ACCOUNT = patch.dict(os.environ, {'WCSS_USER': 'tester', 'WCSS_ACCOUNT': 'tester-account',
                                  'WCSS_GRANT_CPU_HOURS': '500'})
REGISTRATION_DATA = load(REGISTRATION)
EXECUTION = ROOT / 'experiments/rule-compaction/execution.py'
AGENTS = [arm['run_name'] for arm in SHADOW_ARMS]


def setUpModule():
    ACCOUNT.start()


def tearDownModule():
    ACCOUNT.stop()


def registered():
    return approved_registration()


class DerivationTests(unittest.TestCase):
    def test_the_pilot_is_seed_42_on_the_three_tuning_mazes_for_all_seven_arms(self):
        manifest = pilot_manifest(*registered())
        self.assertEqual(manifest['stage'], PILOT_STAGE)
        self.assertEqual(manifest['experiment'], 'rule-compaction')
        self.assertEqual(manifest['seeds'], [42])
        self.assertEqual(manifest['agents'], AGENTS)
        self.assertEqual([entry['id'] for entry in manifest['configurations']], list(TUNING_MAZES))
        self.assertEqual(len(manifest['agents']) * len(manifest['configurations']) * len(manifest['seeds']),
                         REGISTRATION_DATA['cost_pilot']['runs'])
        self.assertEqual(manifest['pilot_allocation_cpu_seconds'], 0)
        self.assertEqual(manifest['partition'], 'lem-cpu-normal')
        self.assertTrue(20 <= manifest['concurrency'] <= 40)

    def test_the_grid_is_seeds_42_to_46_and_carries_the_whole_pilot_reservation(self):
        manifest = grid_manifest(*registered())
        self.assertEqual(manifest['stage'], GRID_STAGE)
        self.assertEqual(manifest['seeds'], list(TUNING_SEEDS))
        self.assertEqual(manifest['agents'], AGENTS)
        self.assertEqual([entry['id'] for entry in manifest['configurations']], list(TUNING_MAZES))
        cost = REGISTRATION_DATA['cost']
        self.assertEqual(manifest['pilot_allocation_cpu_seconds'], cost['pilot_hard_reservation_minutes'] * 60)
        self.assertTrue(20 <= len(manifest['agents']) * len(manifest['seeds']) <= 40)
        self.assertTrue(20 <= manifest['concurrency'] <= 40)

    def test_every_configuration_has_the_registered_limits_pair_registration_and_settings(self):
        pairs = {entry['id']: entry for entry in load(ROOT / 'experiments/maze-baselines/pair-protocol.json')['configurations']}
        for manifest in (pilot_manifest(*registered()), grid_manifest(*registered())):
            for configuration in manifest['configurations']:
                identifier = configuration['id']
                self.assertEqual(configuration['registration'], pairs[identifier])
                self.assertEqual(configuration['evaluation_blocks'], ['validation'])
                self.assertEqual(configuration['protocol'], 'maze-pairs-v1')
                self.assertEqual(configuration['vcp_diagnostics'], 'evaluation_disagreement')
                self.assertEqual(list(configuration['limits_minutes']), AGENTS)
                self.assertEqual(configuration['targets'][-1], configuration['budget_steps'])
                for name, limit in configuration['limits_minutes'].items():
                    archived = REGISTRATION_DATA['cost']['archived_limit_minutes'][identifier][name]
                    self.assertEqual(limit, max(2 * archived, archived + 30))

    def test_specification_paths_hashes_and_temperatures_follow_the_registered_arms(self):
        for manifest in (pilot_manifest(*registered()), grid_manifest(*registered())):
            for configuration in manifest['configurations']:
                self.assertEqual(set(configuration['agent_settings']), set(AGENTS))
                for arm in SHADOW_ARMS:
                    settings = configuration['agent_settings'][arm['run_name']]
                    self.assertEqual(settings['compaction_shadow'], arm['specification'])
                    path = ROOT / settings['compaction_shadow']
                    self.assertEqual(settings['compaction_shadow_sha256'], sha256(path))
                    self.assertEqual(settings.get('temperature'), 24 if arm['run_name'] in ('acs2vcp_k2', 'acs2vcp_k4') else None)
                    expected = 'shadow_without_evidence.json' if arm['run_name'] == 'acs2' else 'shadow_all_operators.json'
                    self.assertEqual(Path(settings['compaction_shadow']).name, expected)

    def test_every_registered_specification_resolves_in_the_checkout_the_way_the_node_resolves_it(self):
        for manifest in (pilot_manifest(*registered()), grid_manifest(*registered())):
            for configuration in manifest['configurations']:
                for name, settings in configuration['agent_settings'].items():
                    with self.subTest(stage=manifest['stage'], maze=configuration['id'], run=name):
                        arguments = collect.shadow_arguments(settings, ROOT)
                        self.assertEqual(arguments, ['--compaction-shadow', str(ROOT / settings['compaction_shadow'])])
                        self.assertTrue(Path(arguments[1]).is_file())

    def test_the_hard_reservation_of_pilot_and_grid_fits_the_ceiling(self):
        pilot, grid = pilot_manifest(*registered()), grid_manifest(*registered())
        ceiling = grid['phase_cpu_hour_limit']
        self.assertEqual(ceiling, REGISTRATION_DATA['cost']['cpu_hour_ceiling'])

        def reservation(manifest):
            batches = ([manifest['configurations']] if manifest['stage'] == PILOT_STAGE
                       else [[configuration] for configuration in manifest['configurations']])
            total = 0
            for batch in batches:
                runs = [(configuration, agent) for configuration in batch for agent in manifest['agents']
                        for _ in manifest['seeds']]
                total += len(runs) * max(configuration['limits_minutes'][agent] for configuration, agent in runs)
            return total * 60
        self.assertLessEqual((reservation(pilot) + reservation(grid)) / 3600, ceiling)
        self.assertGreater((reservation(pilot) + reservation(grid)) / 3600, ceiling - 1)
        self.assertEqual(reservation(pilot), grid['pilot_allocation_cpu_seconds'])

    def test_derivation_through_the_shared_entry_point_returns_the_same_manifests(self):
        registration = Path(REGISTRATION)
        self.assertEqual(derive_manifest(dict(experiment='rule-compaction', stage=PILOT_STAGE)),
                         pilot_manifest(*registered()))
        self.assertEqual(derive_manifest(dict(experiment='rule-compaction', stage=GRID_STAGE), registration),
                         grid_manifest(*registered()))
        self.assertEqual(authorization_manifest(dict(stage=PILOT_STAGE)), pilot_manifest(*registered()))

    def test_the_grid_and_unknown_stages_are_refused_without_the_registration(self):
        for manifest in (dict(stage=GRID_STAGE), dict(stage='approved_cost_pilot'), dict(stage='grid_pending_approval')):
            with self.subTest(stage=manifest['stage']), self.assertRaises(ValueError):
                authorization_manifest(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / 'shadow-protocol.json'
            altered.write_bytes(REGISTRATION.read_bytes() + b' ')
            with self.assertRaises(ValueError):
                authorization_manifest(dict(stage=GRID_STAGE), altered)

    def test_the_derivation_runs_as_a_script_with_the_system_interpreter_contract(self):
        payload = json.dumps(dict(manifest=dict(stage=PILOT_STAGE), registration_path=None))
        result = subprocess.run([sys.executable, '-B', str(EXECUTION), '--derive-manifest'], input=payload,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), pilot_manifest(*registered()))
        refused = subprocess.run([sys.executable, '-B', str(EXECUTION), '--derive-manifest'],
                                 input=json.dumps(dict(manifest=dict(stage=GRID_STAGE), registration_path=None)),
                                 text=True, capture_output=True)
        self.assertNotEqual(refused.returncode, 0)

    def test_the_printed_grid_manifest_names_its_registration_inside_the_clone(self):
        manifest = printed_manifest('grid', '/clone')
        self.assertEqual(manifest['authorization_plan'], '/clone/experiments/rule-compaction/shadow-protocol.json')
        self.assertEqual(manifest['authorization_sha256'], sha256(REGISTRATION))
        self.assertEqual({key: value for key, value in manifest.items()
                          if key not in ('authorization_plan', 'authorization_sha256')},
                         grid_manifest(*registered()))
        self.assertEqual(printed_manifest('pilot', '/clone'), pilot_manifest(*registered()))


class ApprovalTests(unittest.TestCase):
    def test_the_approval_record_carries_the_registration_hash_and_the_ceiling(self):
        approval = load(APPROVAL)
        self.assertIs(approval['approved'], True)
        self.assertEqual(approval['plan_sha256'], sha256(REGISTRATION))
        self.assertEqual(approval['confirmed_on'], '2026-11-19')
        self.assertEqual(approval['phase_cpu_hour_limit'], REGISTRATION_DATA['cost']['cpu_hour_ceiling'])
        self.assertGreater(approval['phase_cpu_hour_limit'], REGISTRATION_DATA['cost']['plan_cpu_hour_expectation'])
        self.assertIs(REGISTRATION_DATA['approved'], True)
        self.assertEqual(REGISTRATION_DATA['approval']['record'], 'approval.json')

    def test_a_registration_or_an_approval_that_does_not_match_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            for changes in (dict(approved=False), dict(plan_sha256='0' * 64), dict(phase_cpu_hour_limit=1)):
                with self.subTest(changes=changes):
                    path = Path(temporary) / 'approval.json'
                    atomic_json(path, dict(load(APPROVAL), **changes))
                    with patch('execution.APPROVAL', path), self.assertRaises(ValueError):
                        approved_registration()
            unapproved = Path(temporary) / 'shadow-protocol.json'
            data = dict(REGISTRATION_DATA, approved=False)
            atomic_json(unapproved, data)
            approval = Path(temporary) / 'approval.json'
            atomic_json(approval, dict(load(APPROVAL), plan_sha256=sha256(unapproved)))
            with patch('execution.REGISTRATION', unapproved), patch('execution.APPROVAL', approval), \
                    self.assertRaises(ValueError):
                approved_registration()

    def test_a_changed_specification_file_is_refused_at_derivation(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy_directory = Path(temporary)
            for arm in SHADOW_ARMS:
                target = copy_directory / arm['specification']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / arm['specification']).read_bytes() + b' ')
            with patch('execution.ROOT', copy_directory), self.assertRaises(ValueError):
                authorization_manifest(dict(stage=PILOT_STAGE))


class SharedPathTests(unittest.TestCase):
    def test_the_pilot_needs_no_separate_approval_and_impostors_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = pilot_manifest(*registered())
            require_authorization(Path(temporary), plan, 'resolved-pilot')
            for field, value in (('agents', ['acs2']), ('seeds', [43]), ('phase_cpu_hour_limit', 6),
                                 ('configurations', plan['configurations'][:-1]), ('memory_mib', 512),
                                 ('unregistered', True)):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    require_authorization(Path(temporary), dict(plan, **{field: value}), 'resolved-pilot')
            for field, value in (('targets', [1000]), ('limits_minutes', {}), ('evaluation_blocks', ['test'])):
                changed = copy.deepcopy(plan)
                changed['configurations'][0][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    require_authorization(Path(temporary), changed, 'resolved-pilot')
            changed = copy.deepcopy(plan)
            changed['configurations'][2]['agent_settings']['acs2vcp_k2']['temperature'] = 12
            with self.assertRaises(ValueError):
                require_authorization(Path(temporary), changed, 'resolved-pilot')
            changed = copy.deepcopy(plan)
            changed['configurations'][1]['agent_settings']['acs2er']['compaction_shadow_sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                require_authorization(Path(temporary), changed, 'resolved-pilot')

    def test_the_grid_requires_the_approval_record_and_rederives_every_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            plan = printed_manifest('grid', ROOT)
            require_authorization(base, plan, 'unused', APPROVAL)
            with self.assertRaises(OSError):
                require_authorization(base, plan, 'unused', None)
            for changes in (dict(approved=False), dict(plan_sha256='0' * 64)):
                refused = base / 'refused.json'
                atomic_json(refused, dict(load(APPROVAL), **changes))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    require_authorization(base, plan, 'unused', refused)
            for field, value in (('seeds', [42]), ('agents', AGENTS[:-1]), ('concurrency', 21),
                                 ('pilot_allocation_cpu_seconds', 0), ('phase_cpu_hour_limit', 6)):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    require_authorization(base, dict(plan, **{field: value}), 'unused', APPROVAL)
            with self.assertRaises(ValueError):
                require_authorization(base, dict(plan, authorization_sha256='0' * 64), 'unused', APPROVAL)
            with self.assertRaises(ValueError):
                require_authorization(base, dict(plan, stage='approved_cost_pilot'), 'unused', APPROVAL)

    def fixtures(self, base):
        binary = base / 'clone/target/x86_64-unknown-linux-musl/release/acs2-measure'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'binary')
        (base / 'clone/fixtures').mkdir()
        atomic_json(base / 'clone/fixtures/goal_measurement_regression.json',
                    dict(configurations=[dict(rows=[dict(preset={'preset': 1})])]))

    def scheduler(self, commands):
        header = 'JobID|JobIDRaw|CPUTimeRAW|TotalCPU|AllocCPUS|ElapsedRaw|MaxRSS|State|ExitCode|NodeList|Partition|Start|End\n'
        identifiers = iter(range(1000, 1010))

        def execute(command):
            commands.append(command)
            if command[0] == 'git':
                return 'a' * 40 if command[-1] == 'HEAD' else ''
            if command[0] == 'sshare':
                return '0'
            if command[0] == 'squeue':
                return ''
            if command[0] == 'sacct':
                return header
            if command[0] == 'sbatch':
                return 'accepted' if '--test-only' in command else str(next(identifiers))
            raise AssertionError(command)
        return execute

    def test_the_pilot_is_one_array_of_21_runs_at_the_registered_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            self.fixtures(base)
            manifest = base / 'pilot-manifest.json'
            atomic_json(manifest, printed_manifest('pilot', base / 'clone'))
            commands = []
            with patch('submit.execute', side_effect=self.scheduler(commands)), contextlib.redirect_stdout(io.StringIO()):
                submit(base / 'pilot', manifest, base / 'clone')
            submitted = [command for command in commands if command[0] == 'sbatch' and '--test-only' not in command]
            self.assertEqual(len(submitted), 1)
            command = submitted[0]
            self.assertIn('--array=0-20%21', command)
            self.assertIn('--time=42', command)
            self.assertIn('--mem=1024M', command)
            self.assertIn('--partition=lem-cpu-normal', command)
            self.assertIn('--job-name=rule-compaction-shadow', command)
            plan = json.loads((base / 'pilot/plan.json').read_text())
            self.assertEqual(plan['measurement_commit'], 'a' * 40)
            ledger = json.loads((base / 'pilot/ledger.json').read_text())
            runs = [(run['configuration_id'], run['agent'], run['seed']) for run in ledger[0]['runs']]
            self.assertEqual(runs, [(maze, agent, 42) for maze in TUNING_MAZES for agent in AGENTS])

    def test_the_grid_is_three_chained_arrays_of_35_runs_inside_the_ceiling(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            self.fixtures(base)
            manifest = base / 'grid-manifest.json'
            atomic_json(manifest, printed_manifest('grid', ROOT))
            commands = []
            with patch('submit.execute', side_effect=self.scheduler(commands)), contextlib.redirect_stdout(io.StringIO()):
                submit(base / 'grid', manifest, base / 'clone', APPROVAL)
            submitted = [command for command in commands if command[0] == 'sbatch' and '--test-only' not in command]
            self.assertEqual([next(part for part in command if part.startswith('--time=')) for command in submitted],
                             ['--time=34', '--time=38', '--time=42'])
            for command in submitted:
                self.assertIn('--array=0-34%35', command)
            self.assertFalse(any(part.startswith('--dependency') for part in submitted[0]))
            self.assertIn('--dependency=afterok:1000', submitted[1])
            self.assertIn('--dependency=afterok:1001', submitted[2])
            ledger = json.loads((base / 'grid/ledger.json').read_text())
            self.assertEqual([entry['configuration_id'] for entry in ledger], list(TUNING_MAZES))
            self.assertEqual({run['seed'] for entry in ledger for run in entry['runs']}, set(TUNING_SEEDS))
            self.assertEqual(sum(len(entry['runs']) for entry in ledger), 105)
            bound = max(entry['bound_seconds'] for entry in ledger) / 3600
            self.assertLessEqual(bound, REGISTRATION_DATA['cost']['cpu_hour_ceiling'])
            self.assertEqual((base / 'grid/approval.json').read_bytes(), APPROVAL.read_bytes())


if __name__ == '__main__':
    unittest.main()
