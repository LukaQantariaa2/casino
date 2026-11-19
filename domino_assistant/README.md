# Domino Assistant (Adjarabet All Fives)

A screen-reading overlay for Adjarabet's 1-vs-1 domino game (All Fives / Muggins, double-six set).
It watches the table, follows every tile, and on your turn recommends a move with your chance of
winning the match. It **never clicks or types** — you play every move yourself.

## Setup and start

1. Run `setup.bat` once. It creates `.venv` and installs numpy, OpenCV, mss and PySide6, plus the
   optional Windows OCR packages that read the bet and the target.
2. Open the domino table in Chrome **fullscreen (F11) at 1920×1080**.
3. Run `run.bat`. The overlay appears at the lower right; drag it anywhere.

Other modes:

- `replay.bat "recording.mp4"` — play a screen recording through the overlay.
- `python run.py --console` — no window, advice in the terminal.
- `python tools\replay_report.py recording.mp4 --advice` — a text transcript of a whole recording.

Hotkeys: **F8** hide/show tips · **F10** reset (forget the round, re-read the table) · **F7** click-through.

## What the overlay shows

- Status. **OK** means it is sure of the position. **UNCERTAIN** means something on screen did not
  fit; it waits and re-reads. **PAUSED** means it kept failing; press F10. **WAITING** means
  between rounds or a message is on screen.
- The recommended move, for example `Play 2|5 on the 5 (left end) +10`. It shows the match-win
  chance, the expected point difference in this round, how many deals were checked (or that the
  endgame was solved exactly), and the next best moves with how much each loses.
- Raise offers ("ოპონენტს სურს ფსონის გაზრდა"): accept or decline, with the match-win chance.
- The match score, target and bet; this round's points; the line with its open ends and count;
  how many tiles the opponent holds; and the boneyard.
- **"opponent has no 3, 5"** — values he cannot hold, learned from his draws and passes.
- A grid of all 28 tiles: yours (blue), played (dark), unseen (could be his or in the boneyard),
  and **boneyard only** (orange, his draws prove he cannot have it).

Every event and every recommendation is written to `logs\session-*.jsonl`.

## Rules as implemented

- 7 tiles each and 14 in the boneyard. You draw automatically until you can play, but the last
  **2** tiles of the boneyard are never drawn; a player who cannot play then passes.
- The first double played is the **spinner**. Its top and bottom open only after both sides are
  covered. While a side of the spinner is uncovered, it counts both its halves.
- After each play, if the open ends add up to a multiple of 5, the player scores that sum. A double
  at an end counts double. The first tile scores too (5|5 gives 10).
- The player who goes out scores the opponent's remaining pips rounded **up** to the next 5
  (2→5, 16→20). This is confirmed in the recordings.
- The match is a race to the target (75 / 175 / 255 / 355). It is decided at the **end of a
  round**: in one recording the opponent reached 180/175 mid-round and play went on.

### Assumptions not yet confirmed by a recording

- **Blocked game** (nobody can play and the boneyard is down to 2): assumed the lower pip total
  wins and scores the other's pips rounded up to 5. This never happened in the recordings.
- **Declining a raise**: assumed to forfeit the match at the current bet, so the advice is
  "accept if the match-win chance is at least 25%" (`ACCEPT_RAISE_FROM` in
  `domino/engine/advisor.py`).
- **Who starts a round**: usually the previous round's winner, but not always, so the advice only
  uses it for the match-win table.

## How it decides

The engine samples the opponent's possible hands, consistent with every draw and pass. In each
sample it plays the round out with a fast look-ahead policy. Once nobody can draw and both hands
are small, it solves the endgame exactly. Each result is converted into a match-win chance with a
table built from 20,000 simulated rounds (`domino/engine/round_outcomes.json`; rebuild it with
`python tools\build_outcomes.py`). In self-play it wins about 63% of rounds against a greedy
player that can see its hand.

## Limits

- The layout is calibrated for 1920×1080 fullscreen (`config/layout/adjarabet_1920x1080.json`).
- In screen *recordings* the mouse cursor can cover pips and cause brief UNCERTAIN messages. Live
  capture does not include the cursor.
- If you start the assistant mid-round, it rebuilds the line from the screen but does not know the
  earlier draws, so the opponent model is weaker until the next round.

## Tests and tools

```
python tests\test_game.py
python tests\test_tracker.py
python tools\selfplay.py 200 0.2        # engine vs. greedy baseline
python tools\raw_obs.py rec.mp4 10 20   # raw screen readings for a time window
```
