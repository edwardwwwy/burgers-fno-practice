from pathlib import Path
import os
import subprocess
import sys


def test_smoke_script_runs_on_cpu() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    result = subprocess.run(
        [sys.executable, "scripts/smoke_test.py"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Smoke test passed." in result.stdout
