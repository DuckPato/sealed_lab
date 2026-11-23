"""Summary of scripts/pool_space_check.py results:  python scripts/pool_space_report.py OUT.json [OUT2.json ...]"""
import json, glob, sys, numpy as np
rows = []
for f in sys.argv[1:]:
    for t in json.load(open(f)):
        for x in t["rows"]:
            rows.append(dict(x, set=t["set"], fmt=t["fmt"]))
ok = [x for x in rows if "q_b" in x]
print("pools", len(rows), "built", len(ok), "errors", [x.get("error") for x in rows if "error" in x][:5])
print("legality: 40 cards in all:", all(x["n_cards"] == 40 for x in ok), "| cards beyond the pool:", sum(x["over_pool"] for x in ok),
      "| max |Ctx.logit - SetModel.logit|: %.2e" % max(x["ctx_vs_model"] for x in ok), "| max |q - evaluate.game_wr|: %.2e" % max(abs(x["q_b"] - x["eval_gwr"]) for x in ok))
print("build seconds: median %.1f, p90 %.1f, max %.1f" % (np.median([x["sec"] for x in ok]), np.quantile([x["sec"] for x in ok], .9), max(x["sec"] for x in ok)))
from collections import Counter
print("builder land counts:", dict(sorted(Counter(x["n_lands"] for x in ok).items())), "| player land counts:", dict(sorted(Counter(x["human_lands"] for x in ok).items())))
g = np.array([x["gap_pp"] for x in ok])
print("builder minus player deck, model score: mean %+.2f pp, median %+.2f, player deck scores higher in %d of %d pools (%.1f%%), worst %.2f pp" % (g.mean(), np.median(g), (g < -1e-9).sum(), len(g), 100 * (g < -1e-9).mean(), g.min()))
bad = [x for x in ok if x["human_bad"] > 0]
print("player decks outside the builder's colour configurations: %d of %d (%.0f%%); by cards outside:" % (len(bad), len(ok), 100 * len(bad) / len(ok)), dict(sorted(Counter(min(x["human_bad"], 6) for x in bad).items())))
ls = [x for x in ok if "ls_minus_builder_pp" in x]
a = np.array([x["ls_minus_builder_pp"] for x in ls])
print("local search from the player's deck (permissive config), %d pools: beats the builder in %d (%.1f%% of all pools); gain where it does: mean %+.2f pp, max %+.2f" % (len(ls), (a > 1e-6).sum(), 100 * (a > 1e-6).sum() / len(ok), a[a > 1e-6].mean() if (a > 1e-6).any() else 0, a.max() if len(a) else 0))
a_bad = np.array([x["ls_minus_builder_pp"] for x in ls if x["human_bad"] > 0])
print("   of the %d outside-configuration decks: local search from them beats the builder in %d (%.0f%%), mean over all of them of max(gain,0): %+.3f pp" % (len(a_bad), (a_bad > 1e-6).sum(), 100 * (a_bad > 1e-6).mean(), np.maximum(a_bad, 0).mean()))
print("   averaged over ALL pools, max(gain, 0): %+.3f pp" % (np.maximum(a, 0).sum() / len(ok)))
fx0 = [x for x in ok if "fix_pp" in x]
print("   sign flip: mean of max(change, 0) over all pools (upper bound for proposing both directions): %+.3f pp" % np.mean([max(x["fix_pp"], 0) for x in fx0]))
fx = [x for x in ok if "fix_pp" in x]
f = np.array([x["fix_pp"] for x in fx])
three = np.array([x["n_colour_basics"] >= 3 or ("+" in x.get("fix_colors", "") and len(x["fix_colors"].split("+")[1]) > 0) for x in fx])
print("\n_deficit sign flipped: %d pools; different deck in %d (%.0f%%); better in %d, worse in %d; mean change %+.3f pp (se %.3f); where different: mean %+.3f, min %+.2f, max %+.2f" % (
    len(f), sum(not x["fix_same"] for x in fx), 100 * np.mean([not x["fix_same"] for x in fx]), (f > 1e-6).sum(), (f < -1e-6).sum(), f.mean(), f.std(ddof=1) / np.sqrt(len(f)),
    f[np.abs(f) > 1e-6].mean() if (np.abs(f) > 1e-6).any() else 0, f.min(), f.max()))
print("   builder decks with basics of 3+ colours: %d of %d; among them mean change %+.3f pp, better %d worse %d" % (sum(x["n_colour_basics"] >= 3 for x in fx), len(fx),
      np.mean([x["fix_pp"] for x in fx if x["n_colour_basics"] >= 3] or [0]), sum(x["fix_pp"] > 1e-6 for x in fx if x["n_colour_basics"] >= 3), sum(x["fix_pp"] < -1e-6 for x in fx if x["n_colour_basics"] >= 3)))
print("\nper set x format: pools | builder-player gap pp | outside-config % | LS-from-player beats builder % | sign-fix mean pp (n different)")
keys = sorted({(x["set"], x["fmt"]) for x in ok})
for s, fm in keys:
    v = [x for x in ok if x["set"] == s and x["fmt"] == fm]
    l = [x.get("ls_minus_builder_pp", -1) for x in v]
    print(f"  {s} {fm}: {len(v)} | {np.mean([x['gap_pp'] for x in v]):+.2f} | {100*np.mean([x['human_bad']>0 for x in v]):.0f}% | {100*np.mean([q>1e-6 for q in l]):.0f}% | {np.mean([x['fix_pp'] for x in v]):+.3f} ({sum(not x['fix_same'] for x in v)}) | builder colours: {dict(Counter(len(x['colors']) + (1 if x['splash'] else 0) for x in v))}")
