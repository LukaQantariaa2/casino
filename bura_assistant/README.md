# Bura Assistant — screen-reading decision support for 5-card Bura (test environment)

A Windows desktop tool that **watches** the Bura table in Chrome, rebuilds the full game state from the screen,
evaluates every legal action, and shows the best one in a small always-on-top overlay.

> **It only observes.** It never clicks, types, presses buttons or talks to the browser in any way.
> You make every move yourself. It was built for internal testing on `testweb.adjarabet.com` with test users.

---

## 1. Install (once)

1. Install **Python 3.10+** for Windows from python.org (tick *Add python.exe to PATH*).
2. Keep this folder next to the card images, like this:

   ```
   Casino\
     cards\            <- the exported 06.webp … 314.webp
     bura_assistant\   <- this folder
   ```

   (Another location works too: set `paths.cards_dir` in `config\app.json`.)
3. Double-click **`setup.bat`**. It creates `.venv` and installs `numpy, opencv-python, mss, PySide6, Pillow`.

## 2. Use it

1. Open the game in **Google Chrome**, go to the table and press **F11** (fullscreen, 1920×1080, browser zoom 100%).
2. Double-click **`run.bat`** — ideally *before* the cards are dealt, so the tool sees the whole round.
3. The overlay sits on the empty left part of the table. Drag it anywhere.

| Hotkey | Action |
|---|---|
| **F8**  | hide / show recommendations |
| **F9**  | open the *Correct game state* dialog |
| **F10** | reset tracking (wait for the next deal) |
| **F7**  | toggle click-through for the overlay |

### What the overlay shows

* **Status** — `OK` (tracking, tips shown), `UNCERTAIN` (something on screen is not explained yet; tips hidden),
  `PAUSED` (state could not be reconciled; tips stopped until you correct it with F9 or a new round starts),
  `WAITING` (no round being tracked).
* **Recommendation** for the current decision:
  * `LEAD K♣` / `LEAD 8♣ Q♣` — your lead,
  * `BEAT 10♣ on 9♣` or `GIVE (face down) 6♠ 7♣` — your answer,
  * `CLAIM — press ვარ` / `CONTINUE — press გაგრძელება` after you win a trick,
  * `ACCEPT (თანხმობა)` / `DECLINE (არ მსურს)` / `COUNTER-RAISE` when the opponent raises,
  * `▲ RAISE (დავი)` hint when your winning chance is high enough.
  Each tip comes with win / loss probabilities and the next-best alternatives.
* **Points** for both players (exact; `+ n?` means n cards in that pile were never seen), trump, deck size,
  opponent hand size, stake.
* **Card map** — all 36 cards: in your hand, held by the opponent (known), in a pile, or still unseen.
* **History** — the last game events (also written to `logs\session-*.jsonl`).

## 3. How it reads the screen

`config\layout\adjarabet_1920x1080.json` holds every region (hand, opponent cards, centre, trump, deck, piles,
buttons, raise dialog, end-of-round panel, avatars) in 1920×1080 coordinates; other 16:9 sizes are scaled.

* **Cards** — two-stage template matching of the card *corner* (rank + suit) against your exported images:
  a fast greyscale pass at half resolution finds candidates, a full-colour pass picks the card, and close
  look-alikes (e.g. J♠/J♣) are separated with the upper half of the whole card. Each card gets a score and a
  margin over the next-best card; low values make the frame untrusted.
* **Trump** — the rotated card under the deck (rotated corner templates).
* **Opponent cards / face-down cards** — the red logo on card backs.
* **Buttons** — ვარ / გაგრძელება / დავი / გატანება by colour and shape; the raise dialog by its green/red buttons.
* **Whose turn** — the timer ring around the avatars.
* **Who won a trick** — the pile badge that changes (a player may give up cards that could beat, so the cards
  alone cannot decide). Claim buttons and who leads next are used as cross-checks.
* **Stake** — the green `X2/X4` badge on the avatar of the player who raised last.
* Only frames where the table is still (not mid-animation) are read, and only regions that changed are re-read.

### Turning observations into events (tracker)

Every change between two still frames must be explained by a game event: new deal, lead, answer, reveal,
trick end, draw, raise, round end. My hand is read directly every frame; hidden parts (opponent hand, deck)
are tracked by counting, with full card-conservation checks (36 cards, no duplicates, sizes, trump position).
A single unexplained frame is ignored (usually a card in motion); repeated ones switch to `UNCERTAIN` and then
`PAUSED`, which **stops all recommendations** until you fix the state (F9) or the next round starts.

## 4. The maths

* **Information set** — everything you know: your hand, the trump card and whether it is still under the deck,
  every card that has been shown, both piles, hand sizes, deck size.
* **Sampling** — the hidden cards are dealt at random many times, consistent with all of the above.
* **Before the deck is empty** — each legal action is played out on every sample with fast, realistic playout
  policies (Perfect-Information Monte Carlo). All actions share the same samples (common random numbers) and
  clearly worse actions are dropped early (successive halving), so the time budget goes to close calls.
* **Once the deck is empty** — every sample is solved **exactly** with minimax + memoisation; when the
  opponent's hand is fully known the answer is exact, with no sampling at all.
* **Claim (ვარ)** — recommended only if P(my pile ≥ win points) ≥ `claim_min_probability` (default 99.9%);
  with every card in your pile seen, it is exact.
* **Raise offer** — accept when `new × (P_win − P_loss) ≥ −current` (declining costs the current stake).
  Without draws the break-even win chance is `(new − current) / (2 × new)`, e.g. 25% for 1→2.
* **Raise hint** — shown when your win probability ≥ `raise_min_win_probability`.

Time per decision: `engine.time_budget_s` in `config\app.json` (default 1.2 s).

## 5. Rules are JSON

`config\rules\bura5_adjarabet.json` (default) and `bura3.json`. Select another file in `config\app.json`
(`paths.rules`) or with `run.bat --rules path\to\rules.json`. Every field:

| Field | Meaning |
|---|---|
| `deck.suits`, `deck.ranks` | the deck (suits D C H S) |
| `rank_order_low_to_high` | beating order (default 6 7 8 9 J Q K 10 A) |
| `card_points` | points per rank |
| `hand_size` | cards per hand (5) |
| `trump.trump_card_is_last_card_of_deck` | the face-up trump card is drawn last |
| `trump.trump_beats_any_non_trump` | any trump beats any side-suit card |
| `lead.min_cards`, `lead.max_cards`, `lead.same_suit_required` | what may be led |
| `response.beating_is_optional` | true = you may give cards instead of beating |
| `response.responder_wins_only_if_all_cards_beaten` | all led cards must be beaten |
| `response.discarded_cards_are_revealed` | given-up cards are shown after the trick |
| `draw.trick_winner_draws_first`, `draw.mode` | draw order (`alternate_one_by_one` or `winner_takes_all_needed_first`) |
| `draw.trick_winner_leads_next` | winner leads the next trick |
| `scoring.win_points` | points needed (61 for 5 cards, 31 for 3 cards) |
| `scoring.claim_enabled`, `claim_only_after_winning_trick` | the ვარ claim |
| `scoring.when_cards_run_out` | `higher_points_wins` or `threshold_or_draw` |
| `scoring.tie_result` | `draw` |
| `scoring.loser_with_zero_tricks_multiplier` | e.g. 2 to count a trickless loss double |
| `stakes.levels` | stake ladder `{value, name}` (1, დავი 2, სე 3, ჩარი 4, ფანჯი 5, შაში 6) |
| `specials` | instant-win hands: `trumps_in_hand`, `same_rank_in_hand`, `same_suit_in_hand` (`enabled`, `count`, `rank`) |
| `assistant.*` | thresholds for claim, raise and counter-raise advice |

## 6. Testing tools

```
python tools\replay_report.py "recording.mp4" --advice   # full pipeline on a recording, text log
replay.bat "recording.mp4"                                # same, with the overlay
python tools\simulate_edge.py --rounds 400               # engine vs a typical player, win rate + 95% CI
python -m unittest discover -s tests                      # unit tests (rules, engine, solver, tracker)
```

`qa\` contains the replay reports of the three test recordings. On those recordings the tracker reproduced
every trick, and its point totals match the end-of-round panels (63 and 61 points). In a 150-round simulation
against a heuristic player, the engine won **72.7% ± 7.1%** of rounds using only 0.04 s per decision.

## 7. Limitations / things to know

* Layout is calibrated for **Chrome fullscreen at 1920×1080, zoom 100%**. For other setups, adjust the region
  coordinates in the layout JSON (or record a video and use `tools\replay_report.py` to check).
* Start before the deal. If you start mid-round the tool waits; use **F9** to enter the state by hand.
* Raises are tracked from the avatar badges (one level per raise). If the stake looks wrong, use *Stake −/+*.
* The overall match score/target is not read; advice is per round.
* The opponent is modelled as a reasonable player, not a specific person.
* The overlay hides itself from screen capture on Windows 10 (2004) and later, so it never covers what is read.
  On older Windows keep it outside the table regions.
