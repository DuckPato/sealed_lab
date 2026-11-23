from __future__ import annotations

import gzip
import shutil

from sealed_lab.ingest.detect import detect_file, expand_paths, fingerprint


def test_detect_named_files(synth_dir):
    g = detect_file(synth_dir / "game_data_public.TST.TradSealed.csv.gz")
    assert (g.kind, g.set_code, g.event, g.fmt) == ("game", "TST", "TradSealed", "bo3")
    g = detect_file(synth_dir / "game_data_public.TST.Sealed.csv.gz")
    assert (g.kind, g.event, g.fmt) == ("game", "Sealed", "bo1")
    g = detect_file(synth_dir / "game_data_public.TST.PremierDraft.csv.gz")
    assert (g.kind, g.event, g.fmt) == ("game", "PremierDraft", "draft")
    d = detect_file(synth_dir / "draft_data_public.TST.PremierDraft.csv.gz")
    assert (d.kind, d.set_code, d.event) == ("draft", "TST", "PremierDraft")
    c = detect_file(synth_dir / "cards.csv")
    assert (c.kind, c.set_code) == ("cards", "TST")
    s = detect_file(synth_dir / "scryfall_TST.json")
    assert (s.kind, s.set_code) == ("scryfall", "TST")
    assert len(g.fingerprint) == 20 and g.size > 0


def test_detect_unknown_files(synth_dir, tmp_path):
    assert detect_file(synth_dir / "truth.json").kind == "unknown"
    assert detect_file(synth_dir / "heldout_pools.json").kind == "unknown"
    p = tmp_path / "random.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    assert detect_file(p).kind == "unknown"
    e = tmp_path / "empty.csv"
    e.write_bytes(b"")
    assert detect_file(e).kind == "unknown"
    assert detect_file(tmp_path / "missing.csv").kind == "unknown"


def test_renamed_files_are_sniffed(synth_dir, tmp_path):
    for src, kind, event in [
        ("game_data_public.TST.TradSealed.csv.gz", "game", "TradSealed"),
        ("game_data_public.TST.Sealed.csv.gz", "game", "Sealed"),
        ("draft_data_public.TST.PremierDraft.csv.gz", "draft", "PremierDraft"),
    ]:
        dst = tmp_path / "my_download (1).csv.gz"
        shutil.copy(synth_dir / src, dst)
        info = detect_file(dst)
        assert info.kind == kind
        assert info.set_code == "TST"
        assert info.event == event
        assert "first data row" in info.note
        dst.unlink()


def test_gz_vs_plain_by_content_not_extension(synth_dir, tmp_path):
    src = synth_dir / "game_data_public.TST.TradSealed.csv.gz"
    plain_bytes = gzip.open(src, "rb").read()
    # plain CSV in a .csv file
    p1 = tmp_path / "game_data_public.TST.TradSealed.csv"
    p1.write_bytes(plain_bytes)
    i1 = detect_file(p1)
    assert (i1.kind, i1.set_code, i1.event) == ("game", "TST", "TradSealed")
    # plain CSV that is mis-named .csv.gz
    p2 = tmp_path / "x" / "game_data_public.TST.TradSealed.csv.gz"
    p2.parent.mkdir()
    p2.write_bytes(plain_bytes)
    assert detect_file(p2).kind == "game"
    # gzip content in a .csv file
    p3 = tmp_path / "y" / "game_data_public.TST.TradSealed.csv"
    p3.parent.mkdir()
    shutil.copy(src, p3)
    assert detect_file(p3).kind == "game"
    # set code upper-cased from name
    p4 = tmp_path / "game_data_public.tst.TradSealed.csv"
    p4.write_bytes(plain_bytes)
    assert detect_file(p4).set_code == "TST"


def test_fingerprint_changes_with_content(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes(b"x" * 3_000_000)
    f1 = fingerprint(p)
    assert fingerprint(p) == f1
    b = bytearray(p.read_bytes())
    b[-10] = 0
    p.write_bytes(bytes(b))
    assert fingerprint(p) != f1


def test_expand_paths_recurses(synth_dir, tmp_path):
    (tmp_path / "sub").mkdir()
    shutil.copy(synth_dir / "cards.csv", tmp_path / "sub" / "cards.csv")
    (tmp_path / "sub" / "notes.txt").write_text("hi", encoding="utf-8")
    got = expand_paths([tmp_path])
    assert [p.name for p in got] == ["cards.csv"]
    assert len(expand_paths([synth_dir])) >= 8


def _replay_csv(path, event="TradSealed"):
    """Minimal replay_data file: same meta + deck_ columns as game data, plus turn-by-turn columns."""
    path.write_text("expansion,event_type,draft_id,won,deck_Bolt,sideboard_Bolt,opening_hand,user_turn_1_cards_drawn,"
                    "oppo_turn_1_cards_drawn\n"
                    f"TST,{event},abc,True,1,0,1|2,1,1\n", encoding="utf-8")
    return path


def test_replay_files_are_not_taken_for_game_data(tmp_path):
    r = detect_file(_replay_csv(tmp_path / "replay_data_public.TST.TradSealed.csv"))
    assert (r.kind, r.set_code, r.event) == ("replay", "TST", "TradSealed")
    renamed = detect_file(_replay_csv(tmp_path / "download.csv"))
    assert renamed.kind == "replay"


def test_registry_migrates_mislabeled_replay_entries(synth_dir, home, tmp_path):
    from sealed_lab.ingest import registry

    rp = _replay_csv(tmp_path / "replay_data_public.TST.TradSealed.csv")
    shutil.copy(synth_dir / "game_data_public.TST.TradSealed.csv.gz", tmp_path / "game_data_public.TST.TradSealed.csv.gz")
    # simulate a registry written by the old detector: the replay file registered as game data
    info = detect_file(rp)
    info.kind = "game"
    info.note = "file name says 'replay' but header looks like 'game'; trusting header"
    assert registry.register(info).registered
    ents = registry.entries("TST")
    kinds = sorted((e["kind"], e["path"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]) for e in ents)
    assert kinds == [("game", "game_data_public.TST.TradSealed.csv.gz"), ("replay", "replay_data_public.TST.TradSealed.csv")]
    assert registry.entries("TST") == ents      # idempotent
