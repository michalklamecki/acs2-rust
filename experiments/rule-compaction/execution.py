import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGISTRATION = HERE / 'shadow-protocol.json'
APPROVAL = HERE / 'approval.json'
PAIR_PROTOCOL = ROOT / 'experiments/maze-baselines/pair-protocol.json'
EXPERIMENT = 'rule-compaction'
PILOT_STAGE = 'authorized_cost_pilot'
GRID_STAGE = 'approved_grid'


def digest_of(content):
    return hashlib.sha256(content).hexdigest()


def approved_registration(registration_path=None):
    content = REGISTRATION.read_bytes()
    if registration_path is not None and Path(registration_path).read_bytes() != content:
        raise ValueError('registration differs from the committed shadow protocol')
    digest = digest_of(content)
    registration = json.loads(content)
    approval = json.loads(APPROVAL.read_text())
    if registration.get('approved') is not True or registration['experiment'] != EXPERIMENT:
        raise ValueError('the shadow protocol is not approved')
    if approval.get('approved') is not True or approval.get('plan_sha256') != digest:
        raise ValueError('the approval record does not carry the hash of the shadow protocol')
    if approval.get('phase_cpu_hour_limit') != registration['cost']['cpu_hour_ceiling']:
        raise ValueError('the approval record differs from the registered CPU-hour ceiling')
    return registration, digest


def verify_inputs(registration):
    if digest_of(PAIR_PROTOCOL.read_bytes()) != registration['pair_protocol_sha256']:
        raise ValueError('the pair protocol differs from the registered one')
    for arm in registration['arms']:
        if digest_of((ROOT / arm['shadow_specification']).read_bytes()) != arm['shadow_specification_sha256']:
            raise ValueError('a shadow specification differs from its registered hash: ' + arm['run_name'])


def configurations(registration):
    pairs = {entry['id']: entry for entry in json.loads(PAIR_PROTOCOL.read_text())['configurations']}
    result = []
    for entry in registration['configurations']:
        configuration = copy.deepcopy(entry)
        configuration['registration'] = pairs[entry['id']]
        result.append(configuration)
    return result


def manifest(registration, digest, stage, seeds, concurrency, pilot_allocation_cpu_seconds):
    execution = registration['execution']
    return dict(stage=stage, experiment=EXPERIMENT, seeds=list(seeds),
                agents=[arm['run_name'] for arm in registration['arms']],
                partition=execution['partition'], cpu_model=execution['cpu_model'],
                memory_mib=execution['memory_mib'], concurrency=concurrency,
                phase_cpu_hour_limit=execution['phase_cpu_hour_limit'],
                first_row_timeout_seconds=execution['first_row_timeout_seconds'],
                pilot_allocation_cpu_seconds=pilot_allocation_cpu_seconds,
                job_name=execution['job_name'], registration_sha256=digest,
                configurations=configurations(registration))


def pilot_manifest(registration, digest):
    pilot = registration['execution']['pilot']
    return manifest(registration, digest, PILOT_STAGE, pilot['seeds'], pilot['concurrency'],
                    pilot['pilot_allocation_cpu_seconds'])


def grid_manifest(registration, digest):
    grid = registration['execution']['grid']
    return manifest(registration, digest, GRID_STAGE, grid['seeds'], grid['concurrency'],
                    grid['pilot_allocation_cpu_seconds'])


def authorization_manifest(manifest, registration_path=None):
    stage = manifest.get('stage')
    if stage == PILOT_STAGE:
        registration, digest = approved_registration()
        verify_inputs(registration)
        return pilot_manifest(registration, digest)
    if stage != GRID_STAGE or registration_path is None:
        raise ValueError('the shadow measurement grid requires its approved registration')
    registration, digest = approved_registration(registration_path)
    verify_inputs(registration)
    return grid_manifest(registration, digest)


def printed_manifest(stage, repository):
    registration, digest = approved_registration()
    verify_inputs(registration)
    if stage == 'pilot':
        return pilot_manifest(registration, digest)
    path = Path(repository).resolve() / 'experiments' / EXPERIMENT / REGISTRATION.name
    return dict(grid_manifest(registration, digest), authorization_plan=str(path), authorization_sha256=digest)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--derive-manifest', action='store_true')
    group.add_argument('--print-manifest', choices=('pilot', 'grid'))
    parser.add_argument('--repo', default=str(ROOT))
    arguments = parser.parse_args()
    if arguments.derive_manifest:
        print(json.dumps(authorization_manifest(**json.load(sys.stdin))))
    else:
        print(json.dumps(printed_manifest(arguments.print_manifest, arguments.repo), indent=2))


if __name__ == '__main__':
    main()
