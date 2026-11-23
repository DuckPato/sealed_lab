"""The hold-out validation cache (model/vcache.py) and the concurrent calculation (model/parallel.py, glm.shared_pool):
a stored result is reused only if EVERY input is unchanged, and reuse never changes a number."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from sealed_lab import config
from sealed_lab.model import glm, parallel, pipeline, validate, vcache
from sealed_lab.types import Cancelled


# ---------------------------------------------------------------------------------------------
# hashing: any change of any input changes the digest
# ---------------------------------------------------------------------------------------------

def test_digest_is_sensitive_to_every_kind_of_input():
    a = np.arange(10.0)
    b = a.copy()
    b[7] += 1e-12
    assert vcache.digest(a) == vcache.digest(a.copy())
    assert vcache.digest(a) != vcache.digest(b)
    assert vcache.digest(a) != vcache.digest(a.astype(np.float32))            # dtype counts
    assert vcache.digest(a) != vcache.digest(a.reshape(2, 5))                 # shape counts
    m = sp.random(20, 20, density=0.2, format="csr", random_state=3)
    m2 = m.copy()
    m2.data[0] += 1e-9
    assert vcache.digest(m) == vcache.digest(m.copy()) and vcache.digest(m) != vcache.digest(m2)
    df = pl.DataFrame({"won": [1, 0, 1], "t": ["a", "b", None], "l": [[1], [2], [3]]})
    assert vcache.digest(df) == vcache.digest(df.clone())
    assert vcache.digest(df) != vcache.digest(df.with_columns(pl.when(pl.int_range(pl.len()) == 1).then(1).otherwise(pl.col("won")).alias("won")))
    assert vcache.digest(df) != vcache.digest(df.rename({"won": "win"}))
    assert vcache.digest({"a": 1, "b": [1, 2]}) != vcache.digest({"a": 1, "b": [1, 3]})
    assert vcache.digest(1) != vcache.digest(1.0) != vcache.digest("1")


def test_space_digest_covers_every_block():
    import types

    base = dict(keys=np.arange(3), phi=np.ones((3, 2)), mu=np.zeros(2), sd=np.ones(2), nb_ids=np.arange(2), col_of_card=np.arange(2),
                D=sp.random(3, 4, density=0.5, format="csr", random_state=1), n_cards_total=2, w=None, attr_names=[], attr_mu=np.zeros(0),
                attr_sd=np.ones(0), attr_load=None, pair_ab=None, arch_ca=None, deck_arch=None,
                regime_phi={"random": np.ones((3, 2))}, cast_cards={"random": sp.eye(3, 2, format="csr")})
    ref = vcache.space_digest(types.SimpleNamespace(**base))
    assert ref == vcache.space_digest(types.SimpleNamespace(**base))
    for key, new in (("phi", np.zeros((3, 2))), ("mu", np.ones(2)), ("sd", np.full(2, 2.0)), ("D", base["D"] * 2),
                     ("regime_phi", {"random": np.zeros((3, 2))}), ("cast_cards", {"random": sp.eye(3, 2, format="csr") * 0.5}),
                     ("pair_ab", np.array([[0, 1]])), ("attr_load", np.ones((2, 1))), ("w", np.ones(3))):
        assert vcache.space_digest(types.SimpleNamespace(**{**base, key: new})) != ref, key


# ---------------------------------------------------------------------------------------------
# the store: keys, hits, misses, damaged entries
# ---------------------------------------------------------------------------------------------

def test_key_covers_code_data_and_settings(tmp_path, monkeypatch):
    c = vcache.ValCache(tmp_path, dict(games="g1", cards="c1", hands="h1"))
    k = c.key("val", "bo3_core", fmt="bo3", seed=0, n_rep=500, space="s1")
    assert k == c.key("val", "bo3_core", fmt="bo3", seed=0, n_rep=500, space="s1")
    assert k != c.key("val", "bo3_core", fmt="bo3", seed=1, n_rep=500, space="s1")           # seed
    assert k != c.key("val", "bo3_core", fmt="bo3", seed=0, n_rep=400, space="s1")           # replicates
    assert k != c.key("val", "bo3_core", fmt="bo3", seed=0, n_rep=500, space="s2")           # deck space
    assert k != c.key("val", "bo1_core", fmt="bo3", seed=0, n_rep=500, space="s1")           # name
    assert k != c.key("rec", "bo3_core", fmt="bo3", seed=0, n_rep=500, space="s1")           # kind
    for base in (dict(games="g2", cards="c1", hands="h1"), dict(games="g1", cards="c2", hands="h1"), dict(games="g1", cards="c1", hands="h2")):
        assert vcache.ValCache(tmp_path, base).key("val", "bo3_core", fmt="bo3", seed=0, n_rep=500, space="s1") != k
    monkeypatch.setattr(vcache, "_code_sig", "another-version-of-the-model-code")           # any edit of the model source
    assert c.key("val", "bo3_core", fmt="bo3", seed=0, n_rep=500, space="s1") != k


def test_code_signature_follows_the_source(tmp_path, monkeypatch):
    sig = vcache.code_signature()
    assert len(sig) == 64
    monkeypatch.setattr(vcache, "_code_sig", None)
    assert vcache.code_signature() == sig                      # deterministic
    # the hash really reads the model sources: a fake module directory changes it
    fake = tmp_path / "model"
    fake.mkdir()
    (fake / "a.py").write_text("x = 1\n")
    (tmp_path / "config.py").write_text("y = 1\n")
    monkeypatch.setattr(vcache, "__file__", str(fake / "vcache.py"))
    monkeypatch.setattr(vcache, "_code_sig", None)
    s1 = vcache.code_signature()
    (fake / "a.py").write_text("x = 2\n")
    monkeypatch.setattr(vcache, "_code_sig", None)
    assert vcache.code_signature() != s1 and s1 != sig


def test_store_roundtrip_and_stale_entries_are_misses(tmp_path):
    c = vcache.ValCache(tmp_path, dict(games="g"))
    key = c.key("val", "bo1_MS", a=1)
    assert c.load("val", "bo1_MS", key) is None
    payload = vcache.normalise(dict(gain=np.float64(0.0123456789012345), lo=-1e-5, n=np.int64(5), nan=float("nan"), arr=np.arange(3)))
    c.save("val", "bo1_MS", key, payload, dict(p=np.linspace(0, 1, 5)))
    got, arrays = c.load("val", "bo1_MS", key)
    assert got["gain"] == 0.0123456789012345 and got["n"] == 5 and got["arr"] == [0, 1, 2] and np.isnan(got["nan"])
    assert got["_created"] and np.allclose(arrays["p"], np.linspace(0, 1, 5))
    # same file name, different key (e.g. the data changed but the first 24 hex digits collided): never served
    path = c._stem("val", "bo1_MS", key).with_suffix(".json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["key"] = "0" * 64
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert c.load("val", "bo1_MS", key) is None
    path.write_text("{not json", encoding="utf-8")                                  # damaged entry: a miss, not a crash
    assert c.load("val", "bo1_MS", key) is None


def test_old_entries_are_pruned(tmp_path):
    c = vcache.ValCache(tmp_path, {})
    for i in range(vcache.KEEP_PER_NAME + 3):
        c.save("val", "bo3_core", c.key("val", "bo3_core", i=i), dict(i=i))
        time.sleep(0.02)
    left = list((tmp_path / "validation").glob("val_bo3_core_*.json"))
    assert len(left) == vcache.KEEP_PER_NAME


# ---------------------------------------------------------------------------------------------
# concurrency helpers
# ---------------------------------------------------------------------------------------------

def test_runner_reports_the_first_real_error_and_aborts_the_rest():
    rn = parallel.Runner(None, max_active=2)
    started = threading.Event()

    def slow():
        started.set()
        while not rn.cancel.is_set():
            time.sleep(0.01)
        raise Cancelled()

    def boom():
        started.wait(2)
        raise ValueError("real error")

    f1, f2 = rn.spawn(slow), rn.spawn(boom)
    with pytest.raises(ValueError, match="real error"):
        rn.result(f1)                       # the waiter of the CANCELLED task still sees the original failure
    with pytest.raises(ValueError):
        rn.result(f2)
    rn.close()


def test_runner_external_cancel():
    ev = threading.Event()
    with parallel.Runner(ev) as rn:
        f = rn.spawn(lambda: time.sleep(5))
        threading.Timer(0.2, ev.set).start()
        with pytest.raises(Cancelled):
            rn.result(f)
        assert rn.cancel.is_set()


def test_progress_mux_is_monotone_and_bounded():
    got = []
    mux = parallel.ProgressMux(lambda f, m: got.append((f, m)), 0.2, 0.6, 2)
    a, b = mux.slot(0), mux.slot(1)
    a(0.5, "x")
    b(1.0, "y")
    a(0.2, "z")                            # a slot never goes backwards
    a(1.0, "w")
    fr = [f for f, _ in got]
    assert fr == sorted(fr) and fr[-1] == pytest.approx(0.6) and fr[0] >= 0.2


def _square(t):
    return t * t


def test_shared_pool_serves_concurrent_callers():
    with glm.shared_pool(2):
        out = {}

        def call(k):
            out[k] = sorted(glm.run_parallel(_square, range(k, k + 5), 99))

        ths = [threading.Thread(target=call, args=(k,)) for k in (0, 10, 20)]
        [t.start() for t in ths]
        [t.join() for t in ths]
    assert out == {k: sorted(i * i for i in range(k, k + 5)) for k in (0, 10, 20)}
    assert glm._SHARED is None
    assert sorted(glm.run_parallel(_square, range(4), 1)) == [0, 1, 4, 9]              # n_jobs=1 runs in-process


def test_shared_pool_cancel_only_drops_own_tasks():
    ev = threading.Event()
    ev.set()
    with glm.shared_pool(2):
        with pytest.raises(Cancelled):
            list(glm.run_parallel(_square, range(5), 2, cancel=ev))
        assert sorted(glm.run_parallel(_square, range(3), 2)) == [0, 1, 4]            # the pool is still usable


# ---------------------------------------------------------------------------------------------
# end to end on the synthetic set: warm runs reuse, any change recomputes, reuse changes no number
# ---------------------------------------------------------------------------------------------

def _strip(d):
    """A validation dict without its timing and cache bookkeeping."""
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k not in ("seconds", "cache", "_created")}
    if isinstance(d, list):
        return [_strip(v) for v in d]
    return d


def _same(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-8, abs=1e-11) or (np.isnan(a) and np.isnan(b)), (path, a, b)
    else:
        assert a == b, (path, a, b)


@pytest.fixture(scope="module")
def warm(calc, tmp_path_factory, monkeypatch_module):
    """A copy of the session's synthetic home (its validation cache holds the cold run's results) and a second calculation in it."""
    home = tmp_path_factory.mktemp("warm_home")
    shutil.copytree(calc["home"], home, dirs_exist_ok=True)
    monkeypatch_module.setenv("SEALED_LAB_HOME", str(home))
    # a priming run first: other tests of the session may have touched this home since the session's cold run (e.g. the castability
    # tables are re-simulated per batch of decks), and a changed input is - correctly - a cache miss
    pipeline.calculate("TST", n_jobs=2, n_boot=4)
    cold = json.loads((config.model_dir("TST") / "report.json").read_text(encoding="utf-8"))
    t0 = time.time()
    meta = pipeline.calculate("TST", n_jobs=2, n_boot=4)
    secs = time.time() - t0
    warm_rep = json.loads((config.model_dir("TST") / "report.json").read_text(encoding="utf-8"))
    return dict(home=home, cold=cold, warm=warm_rep, meta=meta, seconds=secs, cold_seconds=calc["seconds"])


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_warm_run_reuses_every_hold_out_result(warm):
    for fmt, v in warm["warm"]["validation"].items():
        assert v["status"] == "ok"
        parts = v["cache"]["parts"]
        assert parts and all(p["hit"] for p in parts.values()), (fmt, parts)
        assert all(p["created"] for p in parts.values())
    # the report says so
    html = (config.model_dir("TST") / "report.html").read_text(encoding="utf-8")
    assert "Cached from" in html
    assert all(f["cache"]["hit"] for f in warm["warm"]["deployed"]["formats"].values() if f.get("status") == "ok")
    assert all(v["cache"]["hit"] for v in warm["warm"]["recency"]["inner"].values() if v.get("status") == "ok")


def test_reuse_changes_no_number(warm):
    for fmt in warm["cold"]["validation"]:
        _same(_strip(warm["cold"]["validation"][fmt]), _strip(warm["warm"]["validation"][fmt]), fmt)
    _same(_strip(warm["cold"]["deployed"]), _strip(warm["warm"]["deployed"]), "deployed")
    for k in ("recency", "attributes", "crossset", "synergy", "archcond", "matchups", "manasim", "castability"):
        assert warm["cold"][k]["applied"] == warm["warm"][k]["applied"], k
        assert warm["cold"][k]["reason"] == warm["warm"][k]["reason"], k
    cm = {k: v for k, v in json.loads((warm["home"] / "sets" / "TST" / "model" / "meta.json").read_text(encoding="utf-8")).items()}
    assert cm["deployed"]["components"] == warm["cold"]["deployed"]["components"]


def test_warm_run_is_faster(warm):
    assert warm["seconds"] < 0.9 * warm["cold_seconds"], (warm["seconds"], warm["cold_seconds"])


@pytest.fixture(scope="module")
def forced(warm, tmp_path_factory, monkeypatch_module):
    """The matchup layer forced on, so that a deployed combination exists and its test runs (here from cached plain-chain results)."""
    home = tmp_path_factory.mktemp("forced_home")
    shutil.copytree(warm["home"], home, dirs_exist_ok=True)
    monkeypatch_module.setenv("SEALED_LAB_HOME", str(home))
    calls = []
    orig_dep = validate.validate_deployed

    def spy(*a, **k):
        r = orig_dep(*a, **k)
        calls.append((a, k, r))
        return r

    monkeypatch_module.setattr(validate, "validate_deployed", spy)
    pipeline.calculate("TST", n_jobs=2, n_boot=4, matchups="on")
    monkeypatch_module.setattr(validate, "validate_deployed", orig_dep)
    rep = json.loads((config.model_dir("TST") / "report.json").read_text(encoding="utf-8"))
    return dict(home=home, calls=calls, report=rep)


def test_deployed_test_reusing_the_plain_chain_equals_refitting_it(forced):
    """validate_deployed(m_ref=<plain-chain predictions of run_validation / of the cache>) == validate_deployed(m_ref=None, which
    refits the plain chain itself)."""
    assert forced["calls"], "forcing the matchup layer on must run the deployed-combination test"
    a, k, r = forced["calls"][0]
    assert k["m_ref"] is not None and r["status"] == "ok"
    fresh = validate.validate_deployed(*a, **{**k, "m_ref": None, "cache": None})
    for key in ("gain", "lo", "hi", "logloss_M", "logloss_MD", "logloss_B0", "gain_M_vs_B0", "auc_M", "auc_MD", "calibration_slope",
                "calibration_intercept"):
        assert fresh[key] == pytest.approx(r[key], rel=1e-8, abs=1e-10), key


def test_deployed_result_is_cached_under_its_own_key(forced):
    """A second run with the same forced component reuses the deployed test; a different component list does not."""
    dep = forced["report"]["deployed"]["formats"]
    assert dep and all(not o["cache"]["hit"] for o in dep.values() if o.get("status") == "ok")      # first time: computed
    pipeline.calculate("TST", n_jobs=2, n_boot=4, matchups="on")
    dep2 = json.loads((config.model_dir("TST") / "report.json").read_text(encoding="utf-8"))["deployed"]["formats"]
    assert all(o["cache"]["hit"] for o in dep2.values() if o.get("status") == "ok")
    pipeline.calculate("TST", n_jobs=2, n_boot=4, matchups="on", synergy="on")
    dep3 = json.loads((config.model_dir("TST") / "report.json").read_text(encoding="utf-8"))["deployed"]["formats"]
    assert all(not o["cache"]["hit"] for o in dep3.values() if o.get("status") == "ok")


def test_changing_one_setting_recomputes_only_what_depends_on_it(warm, monkeypatch):
    """Synergy switched off: every other component's stored result is reused; the final deployed combination differs, so
    its test is recomputed."""
    monkeypatch.setenv("SEALED_LAB_HOME", str(warm["home"]))
    pipeline.calculate("TST", n_jobs=2, n_boot=4, synergy="off")
    rep = json.loads((config.model_dir("TST") / "report.json").read_text(encoding="utf-8"))
    for fmt, v in rep["validation"].items():
        parts = v["cache"]["parts"]
        assert "MS" not in parts and "synergy" not in v
        assert all(p["hit"] for p in parts.values()), parts
    assert rep["synergy"]["applied"] is False


def test_changed_data_recomputes_everything(warm, monkeypatch, tmp_path):
    """One flipped game result in the corpus => no stored result is served (the key holds the content hash of the games)."""
    home = tmp_path / "home2"
    shutil.copytree(warm["home"], home)
    monkeypatch.setenv("SEALED_LAB_HOME", str(home))
    gp = config.corpus_dir("TST") / "games.parquet"
    g = pl.read_parquet(gp)
    j = int(np.nonzero(g["fmt"].to_numpy() == "bo1")[0][0])
    won = g["won"].to_list()
    won[j] = (not won[j]) if isinstance(won[j], bool) else 1 - won[j]
    g.with_columns(pl.Series("won", won, dtype=g.schema["won"])).write_parquet(gp)
    seen = []
    orig = validate.run_validation

    def spy(*a, **k):
        r = orig(*a, **k)
        seen.append(r.get("cache"))
        return r

    monkeypatch.setattr(validate, "run_validation", spy)
    pipeline.calculate("TST", n_jobs=2, n_boot=4, castability="off", matchups="off", synergy="off", archcond="off",
                       attributes="off", crossset="off", manasim="v1", recency="off")
    assert seen and all(not c["any_hit"] for c in seen), seen


# ---------------------------------------------------------------------------------------------
# memory budget: concurrency bounded by the estimated footprint, out-of-memory tasks retried alone
# ---------------------------------------------------------------------------------------------

def test_memory_status_reads_the_machine():
    from sealed_lab.model import memory

    avail, total = memory.memory_status()
    assert 0 < avail <= total and total > 1 << 28


def test_gate_admits_while_it_fits_and_a_lone_request_always(monkeypatch):
    from sealed_lab.model import memory

    monkeypatch.setattr(memory, "memory_status", lambda: (1 << 40, 1 << 41))
    g = memory.MemoryGate(1000)
    assert g.try_acquire(400) == 400 and g.try_acquire(400) == 400
    assert g.try_acquire(400) is None                       # 1200 > 1000
    g.release(400)
    assert g.try_acquire(400) == 400
    g.release(800)
    assert g.used == 0
    assert g.try_acquire(5000) == 5000                      # larger than the whole budget but alone: runs (sequential fallback)
    assert g.try_acquire(1) is None
    g.release(5000)


def test_gate_blocks_until_memory_is_released(monkeypatch):
    from sealed_lab.model import memory

    monkeypatch.setattr(memory, "memory_status", lambda: (1 << 40, 1 << 41))
    g = memory.MemoryGate(1000)
    g.acquire(900)
    got = []
    t = threading.Thread(target=lambda: got.append(g.acquire(500)))
    t.start()
    time.sleep(0.5)
    assert not got and g.waits == 1
    g.release(900)
    t.join(5)
    assert got == [500]


def test_gate_refuses_when_the_machine_is_short_of_ram(monkeypatch):
    from sealed_lab.model import memory

    g = memory.MemoryGate(10 * memory.GB, floor=memory.GB)
    monkeypatch.setattr(memory, "memory_status", lambda: (3 * memory.GB, 32 * memory.GB))
    assert g.try_acquire(memory.GB) == memory.GB            # alone
    assert g.try_acquire(memory.GB) == memory.GB            # 3 - 1 >= 1 free
    assert g.try_acquire(int(2.5 * memory.GB)) is None             # only 3 GB left of RAM: would not leave the floor free


def test_exclusive_waits_for_the_others_and_admits_its_owner(monkeypatch):
    from sealed_lab.model import memory

    monkeypatch.setattr(memory, "memory_status", lambda: (1 << 40, 1 << 41))
    g = memory.MemoryGate(1000)
    n = g.acquire(300)
    order = []

    def alone():
        with g.exclusive():
            order.append("start")
            assert g.acquire(5000) == 0                     # the owner reserves freely inside
            time.sleep(0.3)
            order.append("end")

    t = threading.Thread(target=alone)
    t.start()
    time.sleep(0.4)
    assert order == []                                      # still waiting for the 300 bytes
    late = threading.Thread(target=lambda: (g.acquire(10), order.append("late")))
    late.start()                                            # new requests queue behind the exclusive one
    time.sleep(0.3)
    g.release(n)
    t.join(5)
    late.join(5)
    assert order == ["start", "end", "late"]


def test_estimates_scale_with_the_data():
    from sealed_lab.model import memory

    assert memory.estimate_chain_bytes(1_000_000) > memory.estimate_chain_bytes(100_000) > 0
    assert memory.estimate_chain_bytes(1000, pairs=True) > memory.estimate_chain_bytes(1000)
    assert memory.estimate_opp_bytes(700_000) > memory.estimate_chain_bytes(700_000)


def test_memory_error_detection():
    from sealed_lab.model import memory

    assert memory.is_memory_error(MemoryError())
    try:
        from numpy._core._exceptions import _ArrayMemoryError
    except ImportError:                                                                  # numpy < 2
        from numpy.core._exceptions import _ArrayMemoryError
    assert memory.is_memory_error(_ArrayMemoryError((10,), np.dtype("f8")))
    try:
        try:
            raise MemoryError()
        except MemoryError as e:
            raise RuntimeError("worker failed") from e
    except RuntimeError as e:
        assert memory.is_memory_error(e)
    assert not memory.is_memory_error(ValueError("x"))


def _timed_task(t):
    """Pool task: sleeps, reports when it ran; task 'oom' raises MemoryError the first time (flag file)."""
    start = time.time()
    flag = t.get("flag")
    if flag and not os.path.exists(flag):
        open(flag, "w").close()
        raise MemoryError("simulated")
    time.sleep(t["sec"])
    return t["i"], start, time.time()


def _sized_tasks(tmp_path, n, sec, mb):
    tasks = []
    for i in range(n):
        p = tmp_path / f"problem_{i}.npz"
        p.write_bytes(b"\0" * (mb << 20))
        tasks.append(dict(path=str(p), fold=0, i=i, sec=sec))
    return tasks


def test_pool_concurrency_follows_the_memory_budget(tmp_path, monkeypatch):
    from sealed_lab.model import memory

    monkeypatch.setattr(memory, "WORKER_BASE", 1 << 20)
    monkeypatch.setattr(memory, "CV_FACTOR", 1.0)
    tasks = _sized_tasks(tmp_path, 6, 0.6, 10)                            # each task "costs" 11 MB
    with memory.budget() as b, glm.shared_pool(4):
        b.pool.budget = int(2.5 * glm.task_cost(tasks[0]))                # room for two at a time
        got = list(glm.run_parallel(_timed_task, tasks, 4))
    assert sorted(i for i, _s, _e in got) == list(range(6))
    spans = [(s, e) for _i, s, e in got]
    most = max(sum(1 for s2, e2 in spans if s2 < e - 0.05 and e2 > s + 0.05) for s, e in spans)
    assert most <= 2, most
    assert b.pool.peak <= b.pool.budget and b.pool.used == 0


def test_pool_task_that_runs_out_of_memory_is_retried_alone(tmp_path, monkeypatch):
    from sealed_lab.model import memory

    monkeypatch.setattr(memory, "WORKER_BASE", 1 << 20)
    tasks = _sized_tasks(tmp_path, 5, 0.5, 1)
    tasks[2]["flag"] = str(tmp_path / "oom.flag")
    with memory.budget() as b, glm.shared_pool(4):
        got = {i: (s, e) for i, s, e in glm.run_parallel(_timed_task, tasks, 4)}
    assert sorted(got) == list(range(5))                                  # nothing was lost
    s2, e2 = got[2]
    assert all(e <= s2 + 0.05 for i, (s, e) in got.items() if i != 2), got    # the retry started after the others had finished
    assert b.pool.used == 0 and b.pool.exclusive_runs == 1


def test_a_second_out_of_memory_is_an_error(tmp_path):
    with glm.shared_pool(2), pytest.raises(Exception) as ei:
        list(glm.run_parallel(_always_oom, [dict(i=0)], 2))
    from sealed_lab.model import memory

    assert memory.is_memory_error(ei.value)


def _always_oom(_t):
    raise MemoryError("still not enough")


def test_runner_retries_an_out_of_memory_task_alone(monkeypatch):
    from sealed_lab.model import memory

    monkeypatch.setattr(memory, "memory_status", lambda: (1 << 40, 1 << 41))
    calls = []
    other_running = threading.Event()
    log = []

    with memory.budget() as b, parallel.Runner(None, max_active=4) as rn:
        def other():
            with rn.slot(100):
                other_running.set()
                time.sleep(0.8)
                log.append("other done")

        def hungry():
            other_running.wait(2)
            with rn.slot(100):
                calls.append(len(calls))
                if len(calls) == 1:
                    raise MemoryError("simulated")
                log.append("retry ran")
                return "ok"

        f1, f2 = rn.spawn(other), rn.spawn(hungry)
        assert rn.result(f2) == "ok"
        rn.result(f1)
    assert calls == [0, 1] and log == ["other done", "retry ran"]
    assert b.main.exclusive_runs == 1 and b.main.used == 0
