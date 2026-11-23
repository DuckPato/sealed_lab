from __future__ import annotations

import gzip
import json

import polars as pl
import pytest

from sealed_lab.ingest import cards as C
from sealed_lab.ingest import scryfall as S


def test_parse_mana_cost():
    p = C.parse_mana_cost("{2}{W}{W}")
    assert p["pips"] == [2, 0, 0, 0, 0] and p["generic"] == 2 and p["mv"] == 4 and p["colors"] == "W" and not p["hybrid"]
    p = C.parse_mana_cost("{1}{U}{B}")
    assert p["pips"] == [0, 1, 1, 0, 0] and p["colors"] == "UB"
    p = C.parse_mana_cost("{1}{R/W}{R/W}")
    assert p["hybrid"] and p["pips"] == [2, 0, 0, 2, 0] and p["generic"] == 1 and p["mv"] == 3 and p["colors"] == "WR"
    p = C.parse_mana_cost("{2/W}{G/P}")
    assert p["hybrid"] and p["pips"][0] == 1 and p["pips"][4] == 1 and p["mv"] == 3
    p = C.parse_mana_cost("{X}{R}{R}")
    assert p["generic"] == 0 and p["mv"] == 2 and p["pips"][3] == 2
    p = C.parse_mana_cost("{C}{S}")
    assert p["generic"] == 2 and p["colors"] == ""
    assert C.parse_mana_cost("")["mv"] == 0 and C.parse_mana_cost(None)["colors"] == ""


def _mini_scryfall():
    return [
        dict(name="Bolt, the Brave", mana_cost="{R}", cmc=1, type_line="Instant", colors=["R"], color_identity=["R"],
             rarity="common", oracle_text="Deal 3", layout="normal", set="tst"),
        dict(name="Fire // Ice", layout="split", mana_cost="{1}{R} // {1}{U}", cmc=4, type_line="Instant // Instant",
             colors=["U", "R"], color_identity=["U", "R"], rarity="uncommon", set="tst",
             card_faces=[dict(name="Fire", mana_cost="{1}{R}", type_line="Instant", oracle_text="a"),
                         dict(name="Ice", mana_cost="{1}{U}", type_line="Instant", oracle_text="b")]),
        dict(name="Delver of Secrets // Insectile Aberration", layout="transform", cmc=1, set="tst", rarity="common",
             type_line="Creature — Human Wizard // Creature — Human Insect", color_identity=["U"],
             card_faces=[dict(name="Delver of Secrets", mana_cost="{U}", type_line="Creature — Human Wizard", oracle_text="x", colors=["U"]),
                         dict(name="Insectile Aberration", mana_cost="", type_line="Creature — Human Insect", oracle_text="y", colors=["U"])]),
        dict(name="Désert Oasis", mana_cost="", cmc=0, type_line="Land", colors=[], color_identity=["G", "W"],
             rarity="rare", produced_mana=["W", "G"], set="tst"),
        dict(name="Plains", mana_cost="", cmc=0, type_line="Basic Land — Plains", rarity="common", produced_mana=["W"], set="tst"),
        dict(name="Prism Rock", mana_cost="{2}", cmc=2, type_line="Artifact", colors=[], color_identity=[], rarity="common",
             produced_mana=["W", "U", "B", "R", "G"], set="tst"),
    ]


def test_cards_table_from_scryfall():
    names = ["Bolt, the Brave", "Fire // Ice", "Delver of Secrets // Insectile Aberration", "Désert Oasis", "Plains",
             "Prism Rock", "Zzz Unknown"]
    df = C.build_cards_table(names, _mini_scryfall())
    assert df["card_id"].to_list() == list(range(len(names)))
    assert df["name"].to_list() == sorted(names)
    r = {x["name"]: x for x in df.iter_rows(named=True)}
    assert r["Bolt, the Brave"]["pip_R"] == 1 and r["Bolt, the Brave"]["colors"] == "R" and r["Bolt, the Brave"]["meta_source"] == "scryfall"
    f = r["Fire // Ice"]
    assert f["colors"] == "UR" and f["hybrid"] and f["pip_R"] == 1 and f["pip_U"] == 1 and f["front_name"] == "Fire"
    d = r["Delver of Secrets // Insectile Aberration"]
    assert d["is_creature"] and d["colors"] == "U" and d["pip_U"] == 1 and d["types"].startswith("Creature - Human Wizard")
    assert r["Désert Oasis"]["is_land"] and r["Désert Oasis"]["produces"] == "WG"
    assert r["Plains"]["is_basic"] and r["Plains"]["rarity"] == "basic" and r["Plains"]["produces"] == "W"
    assert r["Prism Rock"]["produces"] == "WUBRG" and not r["Prism Rock"]["is_land"]
    assert r["Zzz Unknown"]["meta_source"] == "inferred" and not r["Zzz Unknown"]["is_land"]
    assert df.schema["card_id"] == pl.Int32 and df.schema["pip_W"] == pl.Int8


def test_cards_table_17lands_and_inferred_fallbacks():
    csv = {
        "Bolt, the Brave": dict(name="Bolt, the Brave", rarity="common", color_identity="R", mana_value="1", types="Instant", expansion="TST"),
        "Gold Thing": dict(name="Gold Thing", rarity="rare", color_identity="WU", mana_value="4", types="Creature - Sphinx", expansion="TST"),
        "Twin Vale": dict(name="Twin Vale", rarity="common", color_identity="UB", mana_value="0", types="Land", expansion="TST"),
    }
    df = C.build_cards_table(["Bolt, the Brave", "Gold Thing", "Twin Vale", "Mystery", "Island"], None, csv,
                             {"Mystery": {"WB": 100, "WU": 80, "WR": 60, "WG": 50, "UB": 3}})
    r = {x["name"]: x for x in df.iter_rows(named=True)}
    g = r["Gold Thing"]
    assert g["meta_source"] == "17lands" and g["colors"] == "WU" and g["pip_W"] == 1 and g["pip_U"] == 1 and g["generic"] == 2
    assert r["Twin Vale"]["is_land"] and r["Twin Vale"]["produces"] == "UB"
    m = r["Mystery"]
    assert m["meta_source"] == "inferred" and m["colors"] == "W" and m["pip_W"] == 1 and not m["is_land"]
    assert r["Island"]["is_basic"] and r["Island"]["produces"] == "U" and r["Island"]["is_land"]


def test_iter_json_cards_formats(tmp_path):
    cards = _mini_scryfall()
    (tmp_path / "a.json").write_text(json.dumps(cards), encoding="utf-8")
    assert [c["name"] for c in S.iter_json_cards(tmp_path / "a.json")] == [c["name"] for c in cards]
    with gzip.open(tmp_path / "b.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"object": "list", "data": cards}, f)
    assert len(list(S.iter_json_cards(tmp_path / "b.json.gz"))) == len(cards)
    # many objects spanning read chunks
    big = [dict(cards[0], name=f"Card {i}", oracle_text="x" * 500) for i in range(40000)]
    (tmp_path / "c.json").write_text(json.dumps(big), encoding="utf-8")
    assert sum(1 for _ in S.iter_json_cards(tmp_path / "c.json")) == 40000


class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._p = payload
        self.headers = {}

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, cards, per_page=3, fail=None):
        self.cards = cards
        self.per_page = per_page
        self.calls = []
        self.fail = fail

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        if self.fail:
            raise self.fail
        if url.endswith("/cards/search"):
            return self._page(0)
        if "page=" in url:
            return self._page(int(url.split("page=")[1]))
        if url.endswith("/cards/named"):
            nm = params["exact"]
            for c in self.cards:
                if c["name"] == nm or c["name"].split(" // ")[0] == nm:
                    return FakeResp(200, c)
            return FakeResp(404, {})
        return FakeResp(404, {})

    def _page(self, i):
        data = [c for c in self.cards if c.get("set") == "tst"][i * self.per_page:(i + 1) * self.per_page]
        more = (i + 1) * self.per_page < len([c for c in self.cards if c.get("set") == "tst"])
        j = {"data": data, "has_more": more}
        if more:
            j["next_page"] = f"https://api.scryfall.com/cards/search?page={i + 1}"
        return FakeResp(200, j)


def test_scryfall_fetch_paginates_caches_and_goes_offline(home):
    cards = _mini_scryfall()
    bonus = dict(name="Bonus Card", mana_cost="{3}", cmc=3, type_line="Artifact", rarity="special", set="spg")
    sess = FakeSession(cards + [bonus], per_page=2)
    names = [c["name"] for c in cards] + ["Bonus Card", "Nowhere Card"]
    got, warn = S.get_scryfall_cards("TST", names, fetch=True, session=sess, min_interval=0)
    by = {c["name"] for c in got}
    assert {c["name"] for c in cards} <= by and "Bonus Card" in by
    urls = [c[0] for c in sess.calls]
    assert sum(u.endswith("/cards/search") or "page=" in u for u in urls) == 3  # 6 cards / 2 per page
    assert all(c[2]["User-Agent"].startswith("sealed_lab") and c[2]["Accept"] == "application/json" for c in sess.calls)
    assert sess.calls[0][1]["q"] == "e:tst" and sess.calls[0][1]["unique"] == "prints"
    assert any("not found" in w for w in warn)
    # second call: everything cached (incl. not_found) -> no network at all
    sess2 = FakeSession([], fail=RuntimeError("offline"))
    got2, warn2 = S.get_scryfall_cards("TST", names, fetch=True, session=sess2, min_interval=0)
    assert sess2.calls == [] and len(got2) == len(got)


def test_scryfall_offline_and_http_errors_are_not_fatal(home):
    import requests

    for fail in (requests.ConnectionError("no network"), TimeoutError("slow")):
        got, warn = S.get_scryfall_cards("TST", ["Bolt, the Brave"], fetch=True,
                                         session=FakeSession([], fail=fail), min_interval=0)
        assert got == [] and any("unavailable" in w for w in warn)

    class Bad:
        def get(self, *a, **k):
            return FakeResp(403, {})

    got, warn = S.get_scryfall_cards("TST", ["Bolt, the Brave"], fetch=True, session=Bad(), min_interval=0)
    assert got == [] and any("unavailable" in w for w in warn)
    got, warn = S.get_scryfall_cards("TST", ["Bolt, the Brave"], fetch=False)
    assert got == []


def test_scryfall_local_bulk_file(home, tmp_path):
    cards = _mini_scryfall() + [dict(name="Other Set Card", set="abc", type_line="Instant", mana_cost="{U}", cmc=1)]
    p = tmp_path / "bulk.json"
    p.write_text(json.dumps(cards), encoding="utf-8")
    got, warn = S.get_scryfall_cards("TST", ["Bolt, the Brave", "Other Set Card"], local_files=[(str(p), "fp1")], fetch=False)
    names = {c["name"] for c in got}
    assert "Bolt, the Brave" in names and "Other Set Card" in names  # wanted by name even from another set
    # cache remembers the source: file may now disappear
    p.unlink()
    got2, _ = S.get_scryfall_cards("TST", ["Bolt, the Brave"], local_files=[(str(p), "fp1")], fetch=False)
    assert len(got2) == len(got)


def test_name_lookup_variants():
    idx = S.build_index(_mini_scryfall())
    assert S.lookup(idx, "Fire // Ice")["name"] == "Fire // Ice"
    assert S.lookup(idx, "Fire")["name"] == "Fire // Ice"
    assert S.lookup(idx, "Ice")["name"] == "Fire // Ice"
    assert S.lookup(idx, "desert oasis")["name"] == "Désert Oasis"
    assert S.lookup(idx, "Delver of Secrets")["name"].startswith("Delver")
    assert S.lookup(idx, "nothing") is None


def test_printed_name_lookup():
    """Through the Omenpaths (OM1): 17lands uses the printed name of a Spider-Man card; it must resolve, never shadow a real name."""
    cards = [S.trim_card(dict(name="Spectacular Spider-Man", printed_name="Ademi of the Silkchutes", set="om1",
                              type_line="Legendary Creature — Spider Human Hero", rarity="rare", cmc=2)),
             S.trim_card(dict(name="City Pigeon", printed_name="Bolt", set="om1", type_line="Creature — Bird", rarity="common")),
             S.trim_card(dict(name="Bolt", set="tst", type_line="Instant", rarity="common"))]
    assert cards[0]["printed_name"] == "Ademi of the Silkchutes"
    idx = S.build_index(cards)
    assert S.lookup(idx, "Ademi of the Silkchutes")["name"] == "Spectacular Spider-Man"
    assert S.lookup(idx, "ademi of the silkchutes")["name"] == "Spectacular Spider-Man"
    assert S.lookup(idx, "Bolt")["name"] == "Bolt"          # a real name wins over another card's printed name
    t = C.build_cards_table(["Ademi of the Silkchutes"], scryfall_cards=cards)
    assert t["is_creature"][0] and t["rarity"][0] == "rare" and t["meta_source"][0] == "scryfall" and t["mana_value"][0] == 2


def test_lost_character_lookup():
    """17lands writes some non-ASCII letters as '?' (TMT 'Bespoke B?' = 'Bespoke Bō'); only an unambiguous match counts."""
    cards = [S.trim_card(dict(name="Bespoke Bō", set="tmt", type_line="Artifact — Equipment", rarity="uncommon")),
             S.trim_card(dict(name="Cat A", set="tst", type_line="Creature — Cat")),
             S.trim_card(dict(name="Cat B", set="tst", type_line="Creature — Cat"))]
    idx = S.build_index(cards)
    assert S.lookup(idx, "Bespoke B?")["name"] == "Bespoke Bō"
    assert S.lookup(idx, "Cat ?") is None             # two cards fit: ambiguous
    assert S.lookup(idx, "Bespoke ?") is None         # '?' is one character only


def test_lost_character_names_shown_and_exported(monkeypatch):
    cards = [S.trim_card(dict(name="Bespoke Bō", set="tmt", type_line="Artifact — Equipment", rarity="uncommon", cmc=3))]
    t = C.build_cards_table(["Bespoke B?", "Other Card"], scryfall_cards=cards)
    row = t.filter(pl.col("name") == "Bespoke B?")
    assert row["front_name"][0] == "Bespoke Bō" and row["meta_source"][0] == "scryfall" and row["mana_value"][0] == 3
    assert t.filter(pl.col("name") == "Other Card")["front_name"][0] == "Other Card"

    from types import SimpleNamespace
    from sealed_lab import api
    from sealed_lab.deck import builder
    monkeypatch.setattr(builder, "load_model", lambda sc: SimpleNamespace(cards=t))
    assert api._real_names("Deck\n2 Bespoke B?\n1 Other Card\n", "TMT") == "Deck\n2 Bespoke Bō\n1 Other Card\n"
