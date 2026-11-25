# Changes

What each release changed, newest first. The app offers a new release at start and every 3 hours while it is open
(from 0.1.5 on); this page is also the text of each release on GitHub.

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
