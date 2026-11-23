"""Card-attribute prior and cross-set structure prior: card metadata, attribute matrix, design algebra, standard errors,
decision rules, and an end-to-end synthetic calculation with both components switched on."""
from __future__ import annotations

import json
import os

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from sealed_lab.ingest import cards as C
from sealed_lab.ingest import scryfall as S
from sealed_lab.model import attributes as at
from sealed_lab.model import cardvalue, crossset, design, glm, pipeline, structure
from tests.test_cards_scryfall import FakeSession


# ---------------------------------------------------------------------------------------------
# card metadata: power / toughness / keywords, Scryfall cache refresh
# ---------------------------------------------------------------------------------------------

def _cards():
    return [
        dict(name="Sky Knight", mana_cost="{1}{W}", cmc=2, type_line="Legendary Creature — Human Knight", colors=["W"],
             color_identity=["W"], rarity="rare", oracle_text="Flying, first strike\nWhen Sky Knight enters, draw a card.",
             layout="normal", set="tst", power="2", toughness="3", keywords=["Flying", "First strike"]),
        dict(name="Zap", mana_cost="{R}", cmc=1, type_line="Instant", colors=["R"], color_identity=["R"], rarity="common",
             oracle_text="Zap deals 3 damage to any target.", layout="normal", set="tst", keywords=[]),
        dict(name="Star Beast", mana_cost="{3}{G}", cmc=4, type_line="Creature — Beast", colors=["G"], color_identity=["G"],
             rarity="uncommon", oracle_text="Trample (This creature can deal excess damage.)", layout="normal", set="tst",
             power="*", toughness="1+*", keywords=["Trample"]),
        dict(name="Plains", mana_cost="", cmc=0, type_line="Basic Land — Plains", rarity="common", produced_mana=["W"], set="tst",
             keywords=[]),
    ]


def test_cards_table_carries_power_toughness_keywords():
    df = C.build_cards_table([c["name"] for c in _cards()], _cards())
    r = {x["name"]: x for x in df.iter_rows(named=True)}
    assert (r["Sky Knight"]["power"], r["Sky Knight"]["toughness"], r["Sky Knight"]["pt_variable"]) == (2.0, 3.0, False)
    assert r["Sky Knight"]["keywords"] == "first strike,flying"
    assert r["Star Beast"]["pt_variable"] and r["Star Beast"]["power"] == 0.0 and r["Star Beast"]["toughness"] == 1.0
    assert r["Zap"]["power"] is None and r["Zap"]["keywords"] == ""
    # sources without the fields: unknown, not empty
    old = [{k: v for k, v in c.items() if k not in ("power", "toughness", "keywords")} for c in _cards()]
    d2 = C.build_cards_table([c["name"] for c in old], old)
    assert d2["keywords"].null_count() == d2.height and d2["power"].null_count() == d2.height
    inferred = C.build_cards_table(["Mystery"], None)
    assert inferred["keywords"][0] is None and inferred["power"][0] is None


def test_scryfall_cache_without_new_fields_is_refreshed_once(home):
    cards = _cards()
    bonus = dict(name="Bonus Card", mana_cost="{3}", cmc=3, type_line="Artifact Creature — Golem", rarity="special", set="spg",
                 power="3", toughness="3", keywords=["Vigilance"])
    strip = lambda c: {k: v for k, v in c.items() if k not in ("power", "toughness", "keywords")}      # noqa: E731
    old = dict(version=1, set="TST", cards=[strip(c) for c in cards] + [strip(bonus)], sources={}, searched=True, not_found=[])
    S.save_cache("TST", old)
    # drop the "fields" marker the new code writes: this is what a cache from before the change looks like
    j = json.loads(S.cache_path("TST").read_text(encoding="utf-8"))
    j.pop("fields", None)
    S.cache_path("TST").write_text(json.dumps(j), encoding="utf-8")
    names = [c["name"] for c in cards] + ["Bonus Card"]
    # offline: nothing breaks, cards stay as they were, cache stays stale
    import requests

    got, warn = S.get_scryfall_cards("TST", names, fetch=True, session=FakeSession([], fail=requests.ConnectionError("x")),
                                     min_interval=0)
    assert len(got) == len(cards) + 1 and all("keywords" not in c for c in got)
    assert "fields" not in json.loads(S.cache_path("TST").read_text(encoding="utf-8"))
    # online: the set search and the named card are refetched once
    sess = FakeSession(cards + [bonus], per_page=2)
    got, warn = S.get_scryfall_cards("TST", names, fetch=True, session=sess, min_interval=0)
    by = {c["name"]: c for c in got}
    assert by["Sky Knight"]["keywords"] == ["Flying", "First strike"] and by["Sky Knight"]["power"] == "2"
    assert by["Bonus Card"]["power"] == "3" and by["Bonus Card"]["keywords"] == ["Vigilance"]
    assert any(u.endswith("/cards/named") for u, *_ in sess.calls)
    assert json.loads(S.cache_path("TST").read_text(encoding="utf-8"))["fields"] == S.FIELDS_VERSION
    # second call: no network at all
    sess2 = FakeSession([], fail=RuntimeError("offline"))
    S.get_scryfall_cards("TST", names, fetch=True, session=sess2, min_interval=0)
    assert sess2.calls == []


# ---------------------------------------------------------------------------------------------
# attribute matrix
# ---------------------------------------------------------------------------------------------

def test_attribute_matrix_flags():
    cs = _cards() + [
        dict(name="Mind Bolt", mana_cost="{1}{U}{U}", cmc=3, type_line="Instant", colors=["U"], color_identity=["U"], rarity="uncommon",
             oracle_text="Counter target spell. Draw a card.", layout="normal", set="tst", keywords=[]),
        dict(name="Grave Command", mana_cost="{B}", cmc=1, type_line="Sorcery", colors=["B"], color_identity=["B"], rarity="common",
             oracle_text="Destroy target creature an opponent controls.", layout="normal", set="tst", keywords=[]),
        dict(name="Own Flicker", mana_cost="{W}", cmc=1, type_line="Instant", colors=["W"], color_identity=["W"], rarity="common",
             oracle_text="Exile target creature you control, then return it.", layout="normal", set="tst", keywords=[]),
        dict(name="Rally Cry", mana_cost="{2}{W}", cmc=3, type_line="Instant", colors=["W"], color_identity=["W"], rarity="common",
             oracle_text="Create two 1/1 white Soldier creature tokens. Creatures you control get +1/+1 until end of turn.",
             layout="normal", set="tst", keywords=[]),
        dict(name="Elf Scout", mana_cost="{G}", cmc=1, type_line="Creature — Elf", colors=["G"], color_identity=["G"], rarity="common",
             oracle_text="{T}: Add {G}.", layout="normal", set="tst", power="1", toughness="1", keywords=[]),
    ]
    df = C.build_cards_table([c["name"] for c in cs], cs)
    A, names = at.attribute_matrix(df)
    row = {n: dict(zip(names, A[i])) for i, n in enumerate(df.sort("card_id")["name"].to_list())}
    k = row["Sky Knight"]
    assert k["kw_flying"] == 1 and k["kw_first_double_strike"] == 1 and k["kw_trample"] == 0
    assert k["t_creature"] == 1 and k["t_legendary"] == 1 and k["rarity_rare"] == 1 and k["rarity_uncommon"] == 0
    assert (k["power"], k["toughness"], k["mv"], k["n_colors"]) == (2, 3, 2, 1) and k["tag_etb"] == 1 and k["tag_draw"] == 1
    assert row["Zap"]["tag_removal"] == 1 and row["Zap"]["t_instant"] == 1 and row["Zap"]["power"] == 0
    assert row["Star Beast"]["pt_variable"] == 1 and row["Star Beast"]["kw_trample"] == 1
    assert row["Mind Bolt"]["tag_counterspell"] == 1 and row["Mind Bolt"]["tag_draw"] == 1
    assert row["Grave Command"]["tag_removal"] == 1 and row["Own Flicker"]["tag_removal"] == 0
    assert row["Rally Cry"]["tag_token"] == 1 and row["Rally Cry"]["tag_pump_aura_equipment"] == 1
    assert row["Elf Scout"]["tag_fixing"] == 1
    assert not A[df.sort("card_id")["name"].to_list().index("Plains")].any()          # basics carry nothing
    # keywords unknown (old cache): intrinsic keywords are read off the oracle text
    old = [{k_: v for k_, v in c.items() if k_ not in ("power", "toughness", "keywords")} for c in cs]
    A2, _ = at.attribute_matrix(C.build_cards_table([c["name"] for c in old], old))
    r2 = {n: dict(zip(names, A2[i])) for i, n in enumerate(df.sort("card_id")["name"].to_list())}
    assert r2["Sky Knight"]["kw_flying"] == 1 and r2["Sky Knight"]["kw_first_double_strike"] == 1
    assert r2["Star Beast"]["kw_trample"] == 1 and r2["Zap"]["kw_flying"] == 0
    # inferred metadata: only what is known (mana value / colours), no failure
    Ai, _ = at.attribute_matrix(C.build_cards_table(["Mystery"], None, None, {"Mystery": {"WB": 100}}))
    assert Ai.shape == (1, len(names)) and Ai[0, names.index("mv")] > 0 and Ai[0, names.index("tag_removal")] == 0
    # column selection drops attributes carried by too few cards
    sel = at.select_columns(A, np.nonzero(~df.sort("card_id")["is_basic"].to_numpy())[0], min_cards=2)
    assert names.index("mv") in sel and names.index("t_planeswalker") not in sel


# ---------------------------------------------------------------------------------------------
# design algebra and standard errors
# ---------------------------------------------------------------------------------------------

def _toy_space(seed=0, n_decks=300, n_card=12, n_feat=3):
    rng = np.random.default_rng(seed)
    keys = np.arange(n_decks)
    phi = rng.normal(size=(n_decks, n_feat))
    cnt = (rng.random((n_decks, n_card)) < 0.4).astype(int) + (rng.random((n_decks, n_card)) < 0.1)
    rows = [(d, c, int(cnt[d, c])) for d in range(n_decks) for c in range(n_card) if cnt[d, c] > 0]
    dc = pl.DataFrame({"deck_key": [r[0] for r in rows], "card_id": [r[1] for r in rows], "n": [r[2] for r in rows]},
                      schema={"deck_key": pl.Int64, "card_id": pl.Int32, "n": pl.Int8})
    cards = pl.DataFrame({"card_id": np.arange(n_card), "is_basic": np.zeros(n_card, bool)})
    w = rng.random(n_decks) + 0.5
    space = design.build_deck_space(keys, phi, dc, cards, weights=w)
    A = np.column_stack([rng.integers(0, 6, n_card), rng.random(n_card) < 0.3, rng.normal(size=n_card)]).astype(float)
    return space, A, cnt


def test_attribute_columns_and_folding_algebra():
    space, A, cnt = _toy_space()
    sa = design.with_attributes(space, A, ["a", "b", "c"])
    assert sa.n_attr == 3 and sa.D.shape[1] == space.D.shape[1] + 3 and sa.deck_dim == space.n_card + space.n_feat + 3
    Z = sa.D[:, space.n_card + space.n_feat:].toarray()
    assert np.allclose((Z * space.w[:, None]).sum(0), 0, atol=1e-12)                   # centred under the deck weights
    assert np.allclose((Z ** 2 * space.w[:, None]).sum(0), 1, atol=1e-12)              # unit variance
    assert np.allclose(sa.D[:, :space.n_card + space.n_feat].toarray(), space.D.toarray())
    rng = np.random.default_rng(1)
    k = 4
    deck = rng.normal(size=(1, space.n_card + space.n_feat + 3))
    coefs = np.hstack([rng.normal(size=(1, k)), deck])
    ref_row = rng.normal(size=k)
    d = pipeline._draws(coefs, k, sa, ref_row)
    # logit through the design matrix == ref + folded betas + theta (what SetModel / the builder compute)
    direct = (sa.D @ deck[0]) + coefs[0, :k] @ ref_row
    folded = d["ref"][0] + cnt @ d["beta"][0][sa.nb_ids] + sa.D[:, space.n_card:space.n_card + space.n_feat].toarray() @ d["theta"][0]
    assert np.allclose(direct, folded, atol=1e-10)
    # without attribute columns nothing changes
    d0 = pipeline._draws(np.hstack([coefs[:, :k], deck[:, :space.n_card + space.n_feat]]), k, space, ref_row)
    assert np.allclose(d0["beta"][0][space.nb_ids], deck[0, :space.n_card]) and np.isclose(d0["ref"][0], coefs[0, :k] @ ref_row)
    # a constant / absent column is not turned into NaN
    assert with_cols_ok(space, A)


def with_cols_ok(space, A):
    A2 = A.copy()
    A2[:, 0] = 0.0
    s2 = design.with_attributes(space, A2, ["a", "b", "c"])
    return bool(np.isfinite(s2.D.toarray()).all())


def test_cluster_se_contrasts_match_manual_combination():
    rng = np.random.default_rng(3)
    n, k, p, nd = 1500, 3, 5, 120
    C = sp.csr_matrix(np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))]))
    D = sp.csr_matrix(rng.normal(size=(nd, p)) * (rng.random((nd, p)) < 0.6))
    idx = rng.integers(0, nd, size=n).astype(np.int32)
    y = (rng.random(n) < 0.5).astype(float)
    pr = glm.Problem(y=y, C=C, D=D, idx=idx)
    sig = np.full(k + p, 0.7)
    res = glm.fit(pr, np.zeros(k + p), sig)
    grp = np.arange(n) // 10
    L = rng.normal(size=(4, k + p))
    a, b, cl, cr = glm.cluster_se(pr, res.coef, sig, grp, contrasts=L)
    a0, b0 = glm.cluster_se(pr, res.coef, sig, grp)
    assert np.allclose(a, a0) and np.allclose(b, b0)
    # contrast of a unit vector == the coefficient's own SE
    E = np.eye(k + p)[[1, 5]]
    _, _, e_lap, e_rob = glm.cluster_se(pr, res.coef, sig, grp, contrasts=E)
    assert np.allclose(e_lap, a0[[1, 5]]) and np.allclose(e_rob, b0[[1, 5]])
    assert cl.shape == (4,) and np.all(cr > 0)


def test_sigma_spec_vector_layout():
    spec = glm.SigmaSpec(sig_cov=np.ones(2), n_card=3, n_theta=2, kappa=4.0, th_lo=0.1, th_hi=1.0, n_attr=2, attr_sigma=0.3)
    v = spec.vector(0.05)
    assert np.allclose(v, [1, 1, .05, .05, .05, .2, .2, .3, .3])
    spec2 = glm.SigmaSpec(sig_cov=np.ones(2), n_card=3, n_theta=2, th_lo=0.1, th_hi=1.0, th_floor=np.array([0.5, 0.05]))
    assert np.allclose(spec2.vector(0.05)[5:], [0.5, 0.2])         # floor only raises


# ---------------------------------------------------------------------------------------------
# decision rule
# ---------------------------------------------------------------------------------------------

def _val(gain, lo, hi, key="attributes"):
    return {"bo1": dict(status="ok", **{key: dict(gain=gain, lo=lo, hi=hi)})}


def test_decide_component_rules():
    d = pipeline.decide_component
    a = d("auto", "attributes", _val(0.004, 0.001, 0.007), True, "", "the prior")
    assert a["applied"] and "above zero" in a["reason"]
    b = d("auto", "attributes", _val(0.002, -0.001, 0.005), True, "", "the prior")
    assert not b["applied"] and "cannot distinguish" in b["reason"]
    c = d("auto", "attributes", _val(-0.004, -0.007, -0.001), True, "", "the prior")
    assert not c["applied"] and "WORSE" in c["reason"]
    both = {**_val(0.004, 0.001, 0.007), "bo3": dict(status="ok", attributes=dict(gain=-0.01, lo=-0.02, hi=-0.001))}
    assert not d("auto", "attributes", both, True, "", "the prior")["applied"]
    on = d("on", "attributes", _val(-0.004, -0.007, -0.001), True, "", "the prior")
    assert on["applied"] and "forced" in on["reason"]
    assert not d("off", "attributes", _val(0.004, 0.001, 0.007), True, "", "the prior")["applied"]
    na = d("on", "crossset", {}, False, "no other set", "x")
    assert not na["applied"] and na["reason"] == "no other set"
    assert not d("auto", "attributes", {}, True, "", "the prior")["applied"]


# ---------------------------------------------------------------------------------------------
# cross-set prior
# ---------------------------------------------------------------------------------------------

def _fake_set(home, code, theta_by_fmt: dict[str, np.ndarray], sd: np.ndarray, names=None):
    d = home / "sets" / code / "model"
    d.mkdir(parents=True, exist_ok=True)
    names = names or structure.FEATURE_NAMES
    meta = dict(set_code=code, formats=list(theta_by_fmt), feature_names=list(names), phi_sd=sd.tolist(), phi_mean=[0.0] * len(names))
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez(d / "draws.npz", **{f"{f}__theta": np.vstack([t, t + 0.1]) for f, t in theta_by_fmt.items()})


def test_crossset_prior_units_and_spread(home):
    names = structure.FEATURE_NAMES
    n = len(names)
    rng = np.random.default_rng(0)
    sd_a, sd_b, sd_this = rng.uniform(0.5, 3, n), rng.uniform(0.5, 3, n), rng.uniform(0.5, 3, n)
    th_a, th_b = rng.normal(size=n) * 0.1, rng.normal(size=n) * 0.1
    _fake_set(home, "AAA", {"bo3": th_a}, sd_a)
    assert crossset.other_set_effects("AAA", names) == []            # the set itself is never its own prior
    prior = crossset.build_prior(crossset.other_set_effects("TST", names), names, sd_this)
    assert prior["sets"] == ["AAA"] and np.allclose(prior["spread"], 0)
    assert np.allclose(prior["mean"], th_a / sd_a * sd_this)         # std -> raw -> this set's std
    assert np.allclose(prior["raw_mean"], th_a / sd_a)
    _fake_set(home, "BBB", {"bo3": th_b, "bo1": th_b + 0.2}, sd_b)
    ob = {o["set"]: o for o in crossset.other_set_effects("TST", names)}
    assert np.allclose(ob["BBB"]["raw"], (th_b + th_b + 0.2) / 2 / sd_b)       # mean over formats of draw 0
    p2 = crossset.build_prior(list(ob.values()), names, sd_this)
    raw = np.vstack([th_a / sd_a, ob["BBB"]["raw"]])
    assert np.allclose(p2["mean"], raw.mean(0) * sd_this) and np.allclose(p2["spread"], raw.std(0, ddof=1) * sd_this)
    # a feature one set lacks is taken from the other; a missing model never raises
    _fake_set(home, "CCC", {"bo3": th_a[:-1]}, sd_a[:-1], names=names[:-1])
    p3 = crossset.build_prior(crossset.other_set_effects("TST", names), names, sd_this)
    assert np.all(np.isfinite(p3["mean"])) and np.all(np.isfinite(p3["spread"]))
    (home / "sets" / "DDD" / "model").mkdir(parents=True)
    (home / "sets" / "DDD" / "model" / "meta.json").write_text("{not json", encoding="utf-8")
    (home / "sets" / "DDD" / "model" / "draws.npz").write_bytes(b"")
    assert {o["set"] for o in crossset.other_set_effects("TST", names)} == {"AAA", "BBB", "CCC"}
    assert crossset.build_prior([], names, sd_this) is None
    assert "no-op" in crossset.describe(None, "TST")["note"]


def test_theta_floor_raises_structure_sigma_only():
    """The cross-set spread is a floor on the structure sigma of the first stage (never tightens the prior)."""
    spec = cardvalue.make_spec(types_stage(), 4, 3, "draft", theta_floor=np.array([0.9, 0.0, 0.0]))
    v = spec.vector(0.05)
    assert v[spec.sig_cov.shape[0] + 4] == pytest.approx(0.9) and v[spec.sig_cov.shape[0] + 5] == pytest.approx(0.2)


def types_stage():
    import types

    return types.SimpleNamespace(enc=types.SimpleNamespace(sigma=np.ones(2)))


# ---------------------------------------------------------------------------------------------
# end to end on the synthetic set: attributes forced on, a fake second set for the cross-set prior
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def attr_calc(tmp_path_factory):
    from sealed_lab import api
    from sealed_lab.ingest.corpus import build_corpus
    from sealed_lab.synth.generate import generate

    old = os.environ.get("SEALED_LAB_HOME")
    home = tmp_path_factory.mktemp("attr_home")
    synth = tmp_path_factory.mktemp("attr_synth")
    os.environ["SEALED_LAB_HOME"] = str(home)
    try:
        generate(synth, set_code="TST", seed=1, scale=0.2)
        api.add_files([str(synth)])
        build_corpus("TST", n_jobs=2, fetch_scryfall=False)
        rng = np.random.default_rng(5)
        _fake_set(home, "OTH", {"bo3": rng.normal(size=structure.N_FEATURES) * 0.05}, rng.uniform(0.5, 2, structure.N_FEATURES))
        meta = pipeline.calculate("TST", n_jobs=2, n_boot=2, recency="off", attributes="on", crossset="on")
        from sealed_lab.model.artifact import SetModel

        mdir = home / "sets" / "TST" / "model"
        out = dict(meta=meta, model=SetModel.load("TST"), report=json.loads((mdir / "report.json").read_text(encoding="utf-8")),
                   html=(mdir / "report.html").read_text(encoding="utf-8"), cards=pl.read_parquet(mdir / "cards.parquet"),
                   meta_file=json.loads((mdir / "meta.json").read_text(encoding="utf-8")), home=home)
    finally:
        if old is None:
            os.environ.pop("SEALED_LAB_HOME", None)
        else:
            os.environ["SEALED_LAB_HOME"] = old
    return out


def test_attribute_prior_report_and_meta(attr_calc):
    rep, meta = attr_calc["report"], attr_calc["meta_file"]
    a = rep["attributes"]
    assert a["mode"] == "on" and a["applied"] and a["n_attr"] >= 5 and a["gamma_source"] == "final"
    assert meta["attributes"]["applied"] and meta["attributes"]["n_attr"] == a["n_attr"]
    for fmt in ("bo3", "bo1"):
        v = rep["validation"][fmt]
        ma = v["metrics"]["MA"]["vs_M"]["logloss"]
        g = v["attributes"]
        assert g["lo"] <= g["gain"] <= g["hi"] and g["gain"] == pytest.approx(-ma["diff"])
        assert (g["lo"], g["hi"]) == pytest.approx((-ma["hi"], -ma["lo"]))
        assert g["movers"] and g["gamma"] and {"name", "exposure", "moved_pp"} <= set(g["movers"][0])
        assert all(a_["se_pp"] > 0 for a_ in g["gamma"])
        assert g["mean_abs_move_pp_low"] is not None or g["n_low"] == 0
        assert meta["attributes"]["gain"][fmt]["gain"] == pytest.approx(g["gain"])
        assert a["gamma"][fmt] and a["gamma"][fmt][0]["se_pp"] > 0
    html = attr_calc["html"]
    assert "Card-attribute prior" in html and "cards whose value moved most" in html and "Cross-set structure prior" in html


def test_attribute_prior_is_folded_into_exported_card_values(attr_calc):
    model, cards = attr_calc["model"], attr_calc["cards"].sort("card_id")
    nb = ~cards["is_basic"].to_numpy()
    for fmt in ("bo3", "bo1"):
        beta0 = model.card_beta(fmt)
        col = cards[f"beta_{fmt}"].to_numpy()
        assert np.allclose(beta0[nb], col[nb], atol=1e-5)                     # draws.npz and cards.parquet agree (total beta)
        assert (beta0[~nb] == 0).all()
        att = cards[f"beta_attr_{fmt}"].to_numpy()
        assert np.abs(att[nb]).max() > 0 and not np.allclose(att[nb], col[nb])
        assert np.all(cards[f"se_{fmt}"].to_numpy()[nb] > 0)


def test_builder_logit_matches_model_with_attributes(attr_calc):
    from sealed_lab.deck import builder

    model = attr_calc["model"]
    names = model.cards.sort("card_id")["name"].to_list()
    nb = np.nonzero(~model.cards.sort("card_id")["is_basic"].to_numpy())[0]
    lands = [i for i, n in enumerate(names) if n in ("Plains", "Island", "Swamp", "Mountain", "Forest")]
    rng = np.random.default_rng(2)
    for fmt in ("bo3", "bo1"):
        ctx = builder.Ctx(model, fmt)
        for _ in range(6):
            spells = rng.choice(nb, size=23, replace=False)
            deck = {int(i): 1 for i in spells}
            for j in range(17):
                deck[int(lands[j % len(lands)])] = deck.get(int(lands[j % len(lands)]), 0) + 1
            assert ctx.logit(deck) == pytest.approx(model.logit(deck, fmt), abs=1e-9)


def test_crossset_prior_in_calculation(attr_calc):
    xs = attr_calc["report"]["crossset"]
    assert xs["mode"] == "on" and xs["applied"] and xs["prior"]["available"] and xs["prior"]["sets"] == ["OTH"]
    sd = np.array(attr_calc["meta_file"]["phi_sd"])[:len(structure.FEATURE_NAMES)]     # + the dead-cards feature when castability ships
    oth = json.loads((attr_calc["home"] / "sets" / "OTH" / "model" / "meta.json").read_text(encoding="utf-8"))
    th = np.load(attr_calc["home"] / "sets" / "OTH" / "model" / "draws.npz")["bo3__theta"][0]
    expect = th / np.array(oth["phi_sd"]) * sd
    got = np.array([xs["prior"]["mean"][n] for n in structure.FEATURE_NAMES])
    assert np.allclose(got, expect)
    for fmt in ("bo3", "bo1"):
        assert "MX" in attr_calc["report"]["validation"][fmt]["metrics"]
        assert attr_calc["report"]["validation"][fmt]["crossset"]["lo"] <= attr_calc["report"]["validation"][fmt]["crossset"]["hi"]
    assert attr_calc["meta_file"]["crossset"]["sets"] == ["OTH"]


def test_one_set_only_crossset_is_a_noop_and_says_so(tmp_path):
    xs = crossset.describe(crossset.build_prior(crossset.other_set_effects("TST", structure.FEATURE_NAMES, home=tmp_path),
                                                structure.FEATURE_NAMES, np.ones(structure.N_FEATURES)), "TST")
    assert not xs["available"] and "only set" in xs["note"]
    from sealed_lab.model import validate

    html = validate.render_crossset(dict(mode="auto", applied=False, reason="x", prior=xs), {})
    assert "No-op" in html
