"""End-to-end pipeline CLI.

Usage:
    uv run python pipeline.py convert   # xlsx → validated Parquet + quality report
    uv run python pipeline.py fit       # Paris fits + crosscheck + physics audit
    uv run python pipeline.py train     # baselines, split metrics, LOAO
    uv run python pipeline.py predict   # target-alloy predictions + intervals
    uv run python pipeline.py export    # xlsx/csv deliverables + PNG figures
    uv run python pipeline.py all
"""

from __future__ import annotations

import sys

import pyarrow as pa

# Avoid pyarrow's bundled mimalloc: it segfaults intermittently on macOS arm64
# in threaded use (same fix as app/dashboard.py).
pa.set_memory_pool(pa.system_memory_pool())

from afml.pipeline_steps import STEPS, run_all


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        run_all()
    elif arg in STEPS:
        STEPS[arg]()
    else:
        sys.exit(f"unknown step {arg!r}; use one of {list(STEPS)} or 'all'")


if __name__ == "__main__":
    main()
