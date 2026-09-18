"""The first N frames of an extxyz file, for the smoke test's seed data.

    python take_frames.py source.extxyz destination.extxyz 40
"""

import itertools
import sys

from easyal import iter_extxyz, write_extxyz


def main(argv=None) -> int:
    source, dest, count = (argv or sys.argv[1:])[:3]
    n = 0
    with open(dest, "w") as fh:
        for frame in itertools.islice(iter_extxyz(source), int(count)):
            write_extxyz(fh, [frame], append=True)
            n += 1
    print(f"[smoke] seed data: {n} frames from {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
