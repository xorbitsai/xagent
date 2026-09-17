"""Short-lived parser process. Input bytes are snapshotted by the parent."""

import json
import sys

from .defaults import default_registry
from .models import ArtifactContent, ValidationLimits


def main() -> None:
    # Bound decoder allocations on Linux in addition to parent-enforced wall
    # time and per-format expansion limits. macOS does not enforce RLIMIT_AS
    # consistently, so retain the portable input/expansion/pixel budgets there.
    if sys.platform == "linux":
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(
            resource.RLIMIT_AS,
            (
                1024**3 if soft == resource.RLIM_INFINITY else min(soft, 1024**3),
                1024**3 if hard == resource.RLIM_INFINITY else min(hard, 1024**3),
            ),
        )
    limits = ValidationLimits(max_bytes=int(sys.argv[2]))
    # Parser libraries may write warnings to stdout. Keep the protocol output
    # separate from their diagnostics.
    output = sys.stdout
    sys.stdout = sys.stderr
    data = sys.stdin.buffer.read(limits.max_bytes + 1)
    report = default_registry().validate(ArtifactContent(sys.argv[1], data, limits))
    output.write(json.dumps(report.as_dict()))
    output.flush()


if __name__ == "__main__":
    main()
