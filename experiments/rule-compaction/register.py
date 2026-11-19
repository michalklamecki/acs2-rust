import argparse
import sys

from protocol import HERE, dump, intervention_draft, intervention_specifications, load, sha256, shadow_protocol

SHADOW = HERE / 'shadow-protocol.json'
INTERVENTION = HERE / 'intervention-draft.json'
APPROVAL = HERE / 'approval.json'


def outputs():
    result = {INTERVENTION.parent / 'specifications' / name: dump(value)
              for name, value in intervention_specifications().items()}
    result[INTERVENTION] = dump(intervention_draft())
    result[SHADOW] = dump(shadow_protocol())
    return result


def approval_problem():
    if not APPROVAL.exists():
        return 'approval.json is missing'
    approval = load(APPROVAL)
    registration = load(SHADOW)
    if approval.get('approved') is not True or approval.get('plan_sha256') != sha256(SHADOW):
        return 'approval.json does not carry the hash of shadow-protocol.json'
    if approval.get('phase_cpu_hour_limit') != registration['cost']['cpu_hour_ceiling']:
        return 'approval.json carries another CPU-hour ceiling than shadow-protocol.json'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args()
    stale = []
    for path, content in outputs().items():
        if arguments.check:
            if not path.exists() or path.read_text() != content:
                stale.append(path.name)
        else:
            path.write_text(content)
    if stale:
        sys.exit('registration differs from its generator: ' + ', '.join(stale))
    if arguments.check and approval_problem():
        sys.exit(approval_problem())


if __name__ == '__main__':
    main()
