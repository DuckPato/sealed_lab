# Changes

What each release changed, newest first. The app offers a new release at start and every 3 hours while it is open
(from 0.1.5 on); this page is also the text of each release on GitHub.

## 0.1.10 (2026-11-27)

- **Provisional model for a brand-new set**: a set that has only 17Lands pick files (a Premier Draft pick file, no game data yet)
  can now be calculated. Calculate rebuilds every drafted deck and its record from the pick file and learns card values from
  them in about a minute; the other terms (land count, curve, mulligans, skill level) are borrowed from the five newest sets
  that have game data. The Deck builder then works for that set.
  - **Labelled everywhere**: the Sets tab says "Provisional (pick data)", the Deck builder shows a banner, and the model panel
    says what it was built from. In testing against the real Sealed results of five sets the provisional card values predicted
    better than no card values at all in Bo1; they are rougher than a normal model, and win rates read high (by up to about
    4-5 points). Compare decks with each other rather than reading the absolute win rate.
  - **Replaced automatically**: as soon as a game data file is added to the set, the set shows "Recalculation pending" and the next
    Calculate builds the normal model, which replaces the provisional one completely.
  - **Not available yet** for such a set: the Report tab, Deck search and the Pool trainer (they need real games). They say so.
  - Sets that already have game data are not affected in any way.
  - **Ready-made sets** can include a provisional set: it ships and installs as model only (no data tables, no written report).
  - The test behind this, with its rule written beforehand, is `docs/pick_model/`.
- **How it works** (`docs/HOW_IT_WORKS.md`, linked from the README): a plain-language explanation of how games become card
  values, how the builder picks a deck, what the app does not claim, and how changes are tested before they ship.

## 0.1.9 (2026-11-27)

- **Deck builder**:
  - **Rarely played** note: a nonland card opened in at least 50 Sealed pools in its colors and played in under 15% of them
    shows a `rarely played` marker. Its tooltip gives the count ("Played in 8% of the 505 Sealed pools that opened it in its
    colors."). It states how often players chose the card, nothing more: card values and the builder's choices are unchanged.
- **Pool trainer**:
  - Grading uses the model's 16 draws: "Model #1 beats your deck in 11 of 16 draws" instead of a single gap in win-rate points.
  - **Tricky** mode finds pools where the model's deck uses other colors than the pilot's and beats it in at least 7 of 8 draws.
    The model's deck is chosen on the other 8 draws, so the count is a fair check. Finding a pool takes about 10 s per pool checked.
  - Pools you have already practiced are remembered between sessions.
  - **Find similar pools** after Reveal & grade opens Deck search's Similar pools for the pool you practiced.
- **Deck search**:
  - Rebuilds show **Before → After** as the model's score of the first and the rebuilt deck. The game record before and after
    no longer serves as the comparison: runs picked for a bad start rise afterwards by chance.
  - Model scores are kept when a set is copied to another computer, and ready-made sets come with them, so the Model score sort
    is instant.
  - Similar pools show the pilot's model score. The color pairs in Corpus insights are clickable and filter to that pair.
  - Best of 3 sideboarding: an option to show only swaps of cards in your current pool. Swaps are ranked by how often strong
    players (60%+) make them. A post-board win rate is shown only with its game count and a 95% interval.
- **Report**: a new "Opening hands" section lists the cards players most often keep and most often mulligan in their first
  7-card hand. Each card shows its rate, the number of hands and a 95% interval, and appears only if seen in at least 200 hands.
  These are observed choices only: whether a hand is kept depends on all seven cards, and the builder does not use these figures.
  Sets without 17Lands replay data (ECL, FRA) say so.
- **Sets tab**: when a set has only 17Lands pick (draft) files, the greyed-out Calculate button says why. The model needs a game
  data file, and 17Lands often publishes a new set's game data, Sealed above all, some days after its pick data.
- **Ready-made sets** (FRA, HOB, MSH, SOS, TMT): the same models as 0.1.8, now with their Model scores and play rates precomputed.

## 0.1.8 (2026-11-25)

- **Pool trainer** (Deck builder): practice on a real Sealed pool of a strong 17Lands player. You build blind (no values or
  win rates shown), then Reveal & grade compares your deck with the model's best deck and with the player's own build.
- **Deck search**:
  - Color and Arena-rank filters are tick lists: tick any colors (exactly these, or including these) and any ranks.
    17Lands records a rank only for Premier Draft runs.
  - **Model score** sort and column: every real Sealed deck scored by the set's model, the same number the deck scorer gives
    a pasted deck. The first time for a set it takes a minute or two, then it is instant. The colors line of Corpus insights
    shows each color pair's median and best model score.
  - Best of 3 rebuilds now mean a change of the deck a match starts with (sideboarding between games no longer counts).
  - "Similar pools" weighs cards by the model's values; the colors line shows raw win rates next to how strong their
    pilots were (no made-up skill adjustment); "Strong pilots also play" compares strong pilots with decks of the same
    color pairs; "Drawn later" shows its win rate only.
- **Report**: how often each card is splashed (Card explorer column), and lists of often-splashed cards with high and low value.
- **New sets**: when a new set has fewer than 40,000 draft games and you added 17Lands pick files, early picks give the
  card values a better starting point (tested on 10 sets: better Best of 1 predictions, Best of 3 no worse). Older sets are
  unchanged.
- The app says "color" (American spelling) everywhere.
- **Ready-made sets** (FRA, HOB, MSH, SOS, TMT) recalculated with this version; their written reports are refreshed.

## 0.1.7 (2026-11-24)

- **Ready-made sets**: this release carries the maintainer's calculated models of the five newest sets (FRA, HOB, MSH, SOS and TMT).
  They are installed at the next start, so their Deck builder, Report and Deck search work without downloading data or calculating.
  A set you calculated from your own files is kept as it is. The download is larger than usual (about 200 MB).

## 0.1.6 (2026-11-24)

- Every set list shows the newest set first: the Deck builder, Report and Deck search dropdowns were alphabetical, now they follow
  the Sets table (release date, newest on top; sets without a known date last).

## 0.1.5 (2026-11-24)

- **Quick calculation**, on by default (Sets tab checkbox; `calc SET --quick` on the command line): skips four tests that never
  changed a model in 11 sets, so a set calculates in a bit over half the time with the same model.
- The app checks for updates at every start and every 3 hours while it is open (was at most once a day, at start).
- Ready-made sets: a release can carry the maintainer's calculated newest sets, installed at the next start. A set you
  calculated from your own files is never replaced. This release carries none: it raises the size limit for updates so the next
  ones can.
- Fixes from a code review:
  - Castability: adventure, omen and prepare cards counted both halves' mana symbols (a green-and-black adventure looked hard to
    cast in a green deck). Recalculate your sets to get the corrected tables.
  - Lands that search for "a Plains, Swamp, or Forest card" (the Monuments, Farseek) count every colour they find; a few other lands
    read correctly.
  - Mana simulator: a deck with no lands no longer crashes; better London mulligan choice; no drawing past the end of the library.
  - Deck titles name a splash only when a card needs one.
  - The written report's fact checker catches more wrong claims (sign, spelled-out numbers, unchecked "strongest" claims).
  - A cancelled or interrupted calculation can no longer leave a mix of old and new data; two windows can't calculate the same set
    at once.
  - Arena pools: the set in the event name wins over the selected set; blank lines and foil markers in a pasted deck are handled.
  - Update prompt can be opened again after "Later"; closing the app waits for running work; safer updates on Linux.

## 0.1.4 (2026-11-24)

- 17Lands moved its download host: downloads work again.

## 0.1.3 (2026-11-24)

- Updates are signed: the app checks the maintainer's signature before installing, and rolls back if an update fails.
- Update checks are anonymous; the uv download used at first start is pinned and checked.
- The uv download also works when Windows PowerShell is started from PowerShell 7.

## 0.1.2 (2026-11-23)

- Ubuntu support: `run_linux.sh`, in-app updates on Linux, the Arena log found under Steam/Proton, Lutris, Bottles and Wine.
- Layout fixes for wider system fonts.

## 0.1.1 (2026-11-23)

- Release checks on GitHub (no change to the app).

## 0.1.0 (2026-11-23)

- First public release: a 17Lands-based Sealed and Draft deck builder and scorer for Windows, with in-app updates.
