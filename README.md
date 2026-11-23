# sealed_lab

Desktop app that turns 17lands public CSVs into a deck-strength model per set, builds the best Sealed deck from a
pool, and scores a 40-card deck. Primary target: Bo3 Traditional Sealed; Bo1 Sealed also supported.

**Data:** [17Lands public datasets](https://www.17lands.com/public_datasets), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Win rates, scores and reports are this app's own model of that
data; sealed_lab is not affiliated with or endorsed by 17Lands. The app uses only the public datasets: it reads the file
list behind the Public Datasets page (at most every 15 minutes) and downloads the files from 17Lands' public S3 bucket. It
never calls the 17Lands site API (card ratings etc.; see the 17Lands
[usage guidelines](https://www.17lands.com/usage_guidelines)). The credit is shown at the
top right of the main window and at the top of every exported HTML report.

## Install (friends)
You need a GitHub account that the owner has invited to the `DuckPato/sealed_lab` repository (accept the e-mail invite).

1. Download the latest `sealed_lab-X.Y.Z.zip` from the repository's **Releases** page and unzip it anywhere you can write
   to (e.g. `C:\Tools\sealed_lab`; not inside `Program Files`).
2. Double-click `run_windows.bat` once. It installs `uv`, Python 3.12 and the exact dependency versions from `uv.lock` into
   `.venv` (a few minutes, about 1 GB), creates **sealed_lab** shortcuts on the Desktop and in the Start menu, and opens the app.
   Windows SmartScreen may warn about a downloaded script: **More info > Run anyway**.
3. From then on start the app from the shortcut (no console window).
4. **Data:** the app downloads 17Lands' public datasets itself (Sets tab / the panel at the top); nothing from 17Lands is
   bundled. Data and models live in `%LOCALAPPDATA%\sealed_lab` (override with env `SEALED_LAB_HOME`); updates never touch it.
   Messages go to `%LOCALAPPDATA%\sealed_lab\logs\sealed_lab.log`, and unexpected errors also appear in a dialog.

### Updates
The repository is private, so the app has to know who you are to see its releases. Sign in once with the GitHub CLI:

```powershell
winget install GitHub.cli
gh auth login
```

(GitHub.com, HTTPS, "Login with a web browser"). Instead you can paste a token in **Help > GitHub sign-in token…**; a
fine-grained token can't reach another person's repository, so it has to be a classic token with the `repo` scope (which can
read all your private repositories): `gh auth login` is the better choice.

The app looks for a new release at most once a day, a few seconds after it starts (turn it off with **Help > Check for updates
automatically**; check by hand with **Help > Check for updates…**). **Update now** downloads the release, closes the app, copies
the new files over the app folder (`.venv` and your data stay), refreshes the dependencies with `uv sync` and reopens the app.
The update log is `%LOCALAPPDATA%\sealed_lab\logs\update.log`. A copy that is a git checkout is never updated in place.

## Run from source (developer)
`run_windows.bat` works the same from a checkout: `uv sync --frozen --inexact` keeps `.venv` at the versions in `uv.lock`.
`uv sync --frozen --inexact --extra dev` adds pytest; run the tests with `.venv\Scripts\python -m pytest`. Refresh the
dependency versions with `uv lock --upgrade` and run the whole suite before a release.

## Releasing (owner)
1. Raise `__version__` in `sealed_lab/__init__.py` and commit.
2. In the development repository: `python tools/export_public.py --dest <public clone>`. It copies only the shareable files
   (an allowlist in the script) and refuses if any contains a personal string (built-in patterns plus the untracked
   `tools/.export_deny.txt`). In the public clone: commit, `git tag vX.Y.Z`, `git push --follow-tags`.
3. GitHub Actions (`.github/workflows/release.yml`) runs the tests on Windows and publishes `sealed_lab-X.Y.Z.zip` as the
   release; friends' apps offer it on their next start.

CLI: `run_windows_cli.bat status`, `run_windows_cli.bat calc SOS --jobs 16 --boot 16`,
`run_windows_cli.bat build SOS pool.txt --fmt bo3`, `run_windows_cli.bat score SOS deck.txt`.

## Workflow
1. **Sets tab** — the app follows ONE data folder (default: where your 17lands downloader saves, one sub-folder per set code,
   e.g. `HOB\`; shown at the top, **Change…** / **Open** / **Rescan**). 17lands files placed there (any mix of
   `game_data_public.<SET>.<EVENT>.csv.gz`, `draft_data_…`, `replay_data_…`, `cards.csv`) are picked up automatically for the sets
   the app tracks: at start, when the folder changes on disk, when you come back to the window, and on Rescan. Files are read in
   place, never copied; a newer file for the same set+event replaces the old one; a file that disappears is flagged "missing" and
   stays registered (a model already built is not dropped). A set can have its own folder instead of `<data folder>\<SET>`.
   Dropping a folder on the window makes it that set's folder, dropping files registers them.
   One row per set with ONE status: **Ready** (model built from exactly the tracked files), **Recalculation pending** (new or
   changed data, or never used; the tooltip names the files), **Queued**, **Calculating 43%**, **Failed — see details** (the panel
   shows the error and traceback; every failure is also in `logs\errors.log` with time, operation and set). Greyed rows are sets that
   have a folder in the data folder or are on 17lands' list but are not tracked: click one to track its folder or download its files.
   The right-hand panel shows the selected set: games / events / dates per format, replay games, number of cards, when the model was
   built and how long it took, which optional components are on, the hold-out verdict per format against a model without deck
   information, calibration, castability, the deployed combination, the draft → Sealed transfer λ, the best 5 cards per format and
   (collapsed) the tracked files. Useful files per set: game data for **TradSealed** (target), **Sealed**, **PremierDraft**,
   **TradDraft** (priors); replay data feeds the mana-simulator calibration.
2. Press **Calculate** (or **Add to queue** while something runs: sets are calculated one after another; the queue strip lists them
   with ✕ buttons, **Cancel** stops the running one, **Cancel all** also empties the queue; the calculation of a set blocks only that
   set elsewhere in the app). The settings below live in the collapsed **Advanced** group, the log behind **Log**. *Recency weighting* (Auto by default) tests whether weighting recent events more heavily
   predicts newer games better (half-life chosen on older data, confirmed on the hold-out) and uses it only if the gain
   is significant; the report has a section on it either way. *On* forces it, *Off* skips the test. The *Card-attribute
   prior* (Auto by default; CLI `--attributes`) shrinks each card's value towards what its mana value, rarity, types,
   keywords and rules-text tags predict, and is deployed only if the same kind of hold-out test shows a significant gain
   (the borrowed structure prior from other sets' models, CLI `--crossset`, follows the same rule). The *Mana simulator*
   (Auto by default; CLI `--manasim auto|v2|v1`) behind the screw / flood / colour / mana-spent features has a v2 calibrated to real
   play (the hand and draws of each queue incl. Arena's Bo1 hand smoother, the real keep rule, extra cards from draw and
   land-search spells); in Auto the whole model is refitted on it and v2 is used only if that beats v1 on the hold-out.
   The report compares v1 and v2 with the replay data. It ingests (only new/changed files), fetches mana costs from Scryfall (cached; offline
   fallback to `cards.csv`), fits the chain of priors draft → Bo1 Sealed → Bo3 Sealed, runs the bootstrap and the
   held-out validation, and writes `report.html` (**Open report**). Uses all cores by default.
3. **Deck builder** — paste the Arena export of the whole pool (deck + sideboard). The set is detected from the cards ("Set: HOB —
   detected (240/240 cards matched)"; a chooser of the sets with a model appears only when the cards do not tell). The format is
   two radio buttons, Bo1 (Arena Sealed) and Bo3 (Traditional Sealed), preselected to the format with more data for the set (the
   help "?" explains why values are fitted per format). Card names that are not exact are guessed and listed in a yellow box
   ("3 card names were guessed — please confirm") with the five closest names and "Remove this card"; builds use the guesses, and
   changing one marks the result out of date. While the app runs it also watches Arena's log (checkbox "Watch Arena for new sealed pools", on by default; needs Detailed Logs in Arena): when a new Sealed pool is written it is imported at once into an empty pool box, or offered in a bar at the very top of the tab (visible whether or not the pool input is) with **Import** / **Dismiss** when the box already has text (never overwritten without the click). The large card preview under the pool list follows the mouse over the pasted list and
   the deck builder. If the paste is an Arena **deck export with Deck and Sideboard sections**, the Deck is your own build: a first
   tab "Your deck" (fully editable, compared with the best family) comes before the families; **Score only** scores it without
   searching (fast). The old Deck score tab's numbers (score 0-100 with its interval, game / match win rate, expected run wins, P(max
   wins), the run rule, efficiency against the best build, warnings) are in the deck builder's side panel for whatever deck is
   being edited. Returns the **deck families** the model
   cannot tell apart (1 to 5; one tab each, "#1 WU · 73.4% · P(best) 62%"): the search runs under every bootstrap refit of the model,
   similar decks are grouped, and each tab shows the family's best deck, P(best), the gap to #1 with an 80% interval, its core
   cards and its flex slots (marked "flex" on the card, with the alternatives and how often they were chosen). A pool with one
   clear best deck shows one tab. Every tab opens in the **visual deck builder** (MTGO style): card
   images in columns, the deck on top and the rest of the pool below. Group by mana value / colour / type / rarity,
   split creatures from non-creatures, sort, resize the cards, toggle the value badge (pp of game win rate; yellow =
   little data). Double-click, drag & drop or right-click to move cards; Undo / Reset.
   **Basic lands** (unlimited): a strip of the five basics sits beside the sideboard title (double-click one or drag it into
   the deck; drag a deck basic onto the strip, or double-click it in the deck, to take it out), and "Basics: − ● n +"
   buttons above the deck add / remove each colour. Every change is one Undo step and is re-scored at once.
   Every change is re-scored live: game and match win rate, score, change vs the generated build, curve, colour
   pips vs sources, mana simulation. **Sample hand…** (mana simulation section) opens a non-modal window that deals 7-card
   hands from the deck as edited (Bo1 with Arena's hand smoother as the simulator models it, Bo3 plain), with London
   mulligans, draws up to turn 10 and a rough "castable now" line. **Copy Arena export** copies deck + sideboard. "Details (as generated)" keeps
   the list view of the build. Card images come from Scryfall once and are cached in `%LOCALAPPDATA%\sealed_lab\images`.
   When a result arrives the pool input hides itself, so the builder gets the full width: a one-line strip (set, format, card count, **Edit pool**, **Import from Arena**) stands in for it, and while the input is hidden a floating card preview follows the pointer over the cards ("◀ Hide pool input" and dragging the divider still work). Keys: Ctrl+Enter builds, Ctrl+Z / Ctrl+Y undo / redo in the deck builder, Ctrl+Shift+C copies the Arena export, Ctrl+H deals a sample hand; the window remembers its size and position. The side panel keeps the headline
   numbers on top and groups the rest into collapsible sections (open / closed state is remembered; the closed ones
   show a one-line summary). **Hover any number for a plain-language explanation** (what it is, how it is computed,
   how to read it); a card whose mana struggles to cast it (castable on time < 80%) carries a "⚠ 62%" chip, and a deck
   that is unlike real decks (odd land count, ...) gets a "less reliable" caution next to the win rate.
   Side panel extras (each only shows what the data support): **Archetype** (nearest cluster of real decks, its observed
   shrunk win rate with interval, signature cards you do not play), **Synergies** (card pairs that win more together;
   shown only if the sign is supported at P ≥ 90% and the pair is worth ≥ 0.25 pp) and, for Bo3, the **Sideboard plan**:
   per opponent colour pair the swaps that raise the win rate against that opponent by ≥ 0.5 pp with P ≥ 90%, with an
   "SB vs UB" chip on the sideboard cards concerned. Pair synergies (`--synergy`), card values by archetype (`--archcond`)
   and opponent matchups (`--matchups`) are model components tested on the hold-out like the others: when a component
   does not show a significant gain it is not used and the app says so instead of marking anything.
4. **Report tab** — one page per set (or "All sets"): the written **Summary** first (hover a sentence for the figures behind it,
   click it to jump to the section that holds them and flash that section; click may switch Best of 1 / Best of 3 for you; a card
   name in the text shows the card), then the analysis in plain language: the set at a glance, colour pairs, cards where players
   misjudge, archetypes, winning deck shape, how the format changed, Best of 1 against Best of 3 and a note on the numbers. A list of
   sections on the left jumps to one and follows the scrolling. Every estimate carries a **How sure?** badge (Strong / Likely / Weak);
   in the tables the Weak rows are dimmed and folded under "Show N more with no clear difference" (when every row is Weak the three
   biggest effects stay visible). **Export…** saves the page as one HTML file. The Deck builder's "See in Report" link opens the
   colour pair of the deck. **Pool-aware:** when the Report is on the set of the pool pasted in the Deck builder, a **For your pool**
   section follows the Summary (the colour pairs your pool supports with the whole-set result beside each, and the pair the
   builder's best deck uses; your cards that the report calls underrated, overrated or trap rares; your eight best cards) and every
   card you own carries a star (hover it: "In your pool (2 copies)") in the Summary, the Cards section, the archetypes and the
   all-cards table; without a pool for that set a line under the Summary says how to load one. Card values there are points of
   win rate per copy (the model's log-odds effect × 25, as in the report), and a pair's value adds up your best 23 playable cards
   in its colours. **Cheat sheet…** saves one printable page (best colour pairs, the best commons and uncommons of each colour,
   the deck shape to aim for, play or draw, cards to pick up and traps to avoid) and opens it in the browser; it needs no
   internet and fits one A4 or Letter sheet.
(There is no separate Deck score tab any more: see the Deck builder tab above. The written report and the analysis are one Report tab.)

## What the numbers mean
- **Game win rate**: estimated P(win a game) for this deck with an average 17lands pilot, 50/50 play/draw,
  the format's typical opponents; the 80% interval comes from the bootstrap refits of the model (model uncertainty,
  not luck). The scale is not centred on 50%: the average real deck of the format is shown next to it ("avg real deck").
  Match win rate (Bo3) = p²(3−2p).
- **Score 0–100**: percentile of that win rate among all decks 17lands users registered for this set and format.
  50 = median 17lands deck (17lands users are stronger than average). Scores compare within one set only.
- **Value (pp)**: change in game win rate from taking one copy of the card out (deck) / adding one (sideboard), holding
  the rest; "above replacement" compares with the best sideboard replacement instead.
- **P(better) / P(best)**: the share of the model's bootstrap refits (16 by default; the tooltip says "15/16 refits") in which
  a swap helps / a family's deck is the best. They step in 1/16 and are coarse, not probabilities of being right.
- **Win-rate breakdown**: percentage points of game win rate (the model's log-odds are in the tooltips), applied from
  50% one bar after the other.
- **Castable on time / less reliable**: from the mana simulation / the deck being outside the range of real decks.
- **low data**: the card has little play in the target format; its value leans on the Bo1/draft estimate.
- The report states, out of sample, whether the model beats simple baselines (mean GIH WR of the deck) and how
  well calibrated it is. Read it after every Calculate on a new set.

## Development
`python -m pytest -q` (Linux or Windows). `python -m sealed_lab cli synth OUTDIR --scale 0.3` writes a synthetic
set in exact 17lands format with known card strengths, used by the tests. `SEALED_LAB_FAKE_API=1 python -m
sealed_lab` runs the GUI on fake data. Design: `SPEC.md`.

## Layout
All panes are resizable: drag the dividers (they highlight on hover); sizes are remembered between sessions. Table
columns can be dragged wider (double-click a divider to fit the contents); the Sets tab shows the full path and note
of the selected file in the pane below the file list.
