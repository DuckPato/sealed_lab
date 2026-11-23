from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sealed_lab import cli

ROOT = Path(__file__).resolve().parents[1]


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def test_cli_synth_add_status_calc(home, tmp_path, capsys):
    out = tmp_path / "syn"
    assert cli.main(["synth", str(out), "--scale", "0.05"]) == 0
    assert "TradSealed" in capsys.readouterr().out
    assert cli.main(["status"]) == 0
    assert "No sets registered" in capsys.readouterr().out
    assert cli.main(["add", str(out)]) == 0
    txt = capsys.readouterr().out
    assert "game_data_public.TST.TradSealed.csv.gz" in txt and "recognised" in txt
    assert cli.main(["status"]) == 0
    txt = capsys.readouterr().out
    assert "Set TST" in txt and "corpus=no" in txt
    if not _has("sealed_lab.model.pipeline"):
        assert cli.main(["calc", "TST", "--jobs", "2", "--no-scryfall"]) == 0
        txt = capsys.readouterr().out
        assert "corpus=yes" in txt and "bo3=" in txt and "model not implemented yet" in txt
        assert cli.main(["status", "TST"]) == 0


def test_python_dash_m_entrypoint(tmp_path):
    env = dict(os.environ, SEALED_LAB_HOME=str(tmp_path / "h"), PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-m", "sealed_lab", "cli", "status"], capture_output=True, text=True, env=env,
                       cwd=str(tmp_path), timeout=120)
    assert r.returncode == 0, r.stderr
    assert "No sets registered" in r.stdout


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])
