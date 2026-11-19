import argparse
import sys

from protocol import HERE, dump, intervention_draft, intervention_specifications, shadow_protocol

SHADOW = HERE / 'shadow-protocol.json'
INTERVENTION = HERE / 'intervention-draft.json'


def outputs():
    result = {INTERVENTION.parent / 'specifications' / name: dump(value)
              for name, value in intervention_specifications().items()}
    result[INTERVENTION] = dump(intervention_draft())
    result[SHADOW] = dump(shadow_protocol())
    return result


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


if __name__ == '__main__':
    main()
