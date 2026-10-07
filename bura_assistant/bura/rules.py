"""Rule configuration loaded from JSON.

Every game-rule decision in the engine, tracker and recognizer reads from a
``Rules`` instance, so a different Bura variant only needs another JSON file.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "name": "Bura",
    "description": "",
    "deck": {"suits": ["D", "C", "H", "S"], "ranks": ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]},
    "rank_order_low_to_high": ["6", "7", "8", "9", "J", "Q", "K", "10", "A"],
    "card_points": {"6": 0, "7": 0, "8": 0, "9": 0, "10": 10, "J": 2, "Q": 3, "K": 4, "A": 11},
    "hand_size": 5,
    "trump": {"trump_card_is_last_card_of_deck": True, "trump_beats_any_non_trump": True},
    "lead": {"min_cards": 1, "max_cards": 5, "same_suit_required": True,
             "allow_mixed_suits_if_all_trumps": False},
    "response": {"must_play_same_number_of_cards": True, "beating_is_optional": True,
                 "responder_wins_only_if_all_cards_beaten": True, "discarded_cards_are_revealed": True},
    "draw": {"trick_winner_draws_first": True, "trick_winner_leads_next": True,
             "mode": "alternate_one_by_one"},
    "scoring": {"win_points": 61, "total_points_in_deck_hint": 120, "claim_enabled": True,
                "claim_only_after_winning_trick": True, "false_claim_result": "lose_round",
                "when_cards_run_out": "higher_points_wins", "tie_result": "draw",
                "loser_with_zero_tricks_multiplier": 1},
    "stakes": {"levels": [{"value": 1, "name": "1"}, {"value": 2, "name": "2"}],
               "raise_allowed_only_on_own_turn": True,
               "same_player_cannot_raise_twice_in_a_row": True,
               "declining_a_raise_loses": "current_stake", "fold_loses": "current_stake"},
    "specials": [],
    "assistant": {"claim_min_probability": 0.999, "raise_min_win_probability": 0.70,
                  "counter_raise_min_win_probability": 0.80},
}

SUIT_SYMBOLS = {"D": "♦", "C": "♣", "H": "♥", "S": "♠"}
SYMBOL_TO_SUIT = {v: k for k, v in SUIT_SYMBOLS.items()}


class RulesError(ValueError):
    pass


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class Special:
    name: str
    ctype: str
    count: int
    rank: str | None
    check: str
    effect: str


@dataclass
class Rules:
    raw: dict
    name: str
    suits: list[str]
    ranks: list[str]
    n_cards: int
    hand_size: int
    win_points: int
    # per-card lookup tables, indexed by card id
    suit_of: list[int] = field(default_factory=list)
    rank_of: list[int] = field(default_factory=list)
    strength: list[int] = field(default_factory=list)
    points: list[int] = field(default_factory=list)
    specials: list[Special] = field(default_factory=list)

    # ---------- construction ----------
    @classmethod
    def from_dict(cls, data: dict) -> "Rules":
        raw = _merge(DEFAULTS, data)
        suits = list(raw["deck"]["suits"])
        ranks = list(raw["deck"]["ranks"])
        order = list(raw["rank_order_low_to_high"])
        pts = raw["card_points"]
        if sorted(order) != sorted(ranks):
            raise RulesError("rank_order_low_to_high must contain exactly the deck ranks")
        for r in ranks:
            if r not in pts:
                raise RulesError(f"card_points is missing rank {r}")
        for s in suits:
            if s not in SUIT_SYMBOLS:
                raise RulesError(f"unknown suit code {s}; use D, C, H, S")
        hand = int(raw["hand_size"])
        if hand < 1 or 2 * hand + 1 > len(suits) * len(ranks):
            raise RulesError("hand_size does not fit the deck")
        lead = raw["lead"]
        if not (1 <= lead["min_cards"] <= lead["max_cards"] <= hand):
            raise RulesError("lead.min_cards/max_cards must satisfy 1 <= min <= max <= hand_size")
        levels = raw["stakes"]["levels"]
        if not levels or any("value" not in lv for lv in levels):
            raise RulesError("stakes.levels must be a non-empty list of {value, name}")

        r = cls(raw=raw, name=raw["name"], suits=suits, ranks=ranks,
                n_cards=len(suits) * len(ranks), hand_size=hand,
                win_points=int(raw["scoring"]["win_points"]))
        for si, _s in enumerate(suits):
            for ri, rk in enumerate(ranks):
                r.suit_of.append(si)
                r.rank_of.append(ri)
                r.strength.append(order.index(rk))
                r.points.append(int(pts[rk]))
        for sp in raw.get("specials", []):
            if not sp.get("enabled", True):
                continue
            cond = sp.get("condition", {})
            if cond.get("type") not in ("trumps_in_hand", "same_rank_in_hand", "same_suit_in_hand"):
                raise RulesError(f"unsupported special condition {cond.get('type')}")
            if sp.get("effect", "win_round") != "win_round":
                raise RulesError("only effect 'win_round' is supported for specials")
            r.specials.append(Special(sp.get("name", "special"), cond["type"], int(cond.get("count", hand)),
                                      cond.get("rank"), sp.get("check", "after_every_draw"),
                                      sp.get("effect", "win_round")))
        return r

    @classmethod
    def load(cls, path: str | Path) -> "Rules":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # ---------- convenience accessors ----------
    def opt(self, section: str, key: str):
        return self.raw[section][key]

    @property
    def total_points(self) -> int:
        return sum(self.points)

    @property
    def stake_levels(self) -> list[dict]:
        return self.raw["stakes"]["levels"]

    def stake_name(self, value: int) -> str:
        for lv in self.stake_levels:
            if lv["value"] == value:
                return str(lv.get("name", value))
        return str(value)

    def next_stake(self, value: int) -> int | None:
        vals = sorted(lv["value"] for lv in self.stake_levels)
        for v in vals:
            if v > value:
                return v
        return None

    def assistant(self, key: str):
        return self.raw["assistant"][key]

    # ---------- card helpers ----------
    def card_id(self, rank: str, suit: str) -> int:
        return self.suits.index(suit) * len(self.ranks) + self.ranks.index(rank)

    def card_name(self, c: int) -> str:
        return f"{self.ranks[self.rank_of[c]]}{SUIT_SYMBOLS[self.suits[self.suit_of[c]]]}"

    def parse_card(self, text: str) -> int:
        t = text.strip().upper().replace("♦", "D").replace("♣", "C").replace("♥", "H").replace("♠", "S")
        if t.startswith("T"):
            t = "10" + t[1:]
        if len(t) < 2:
            raise ValueError(f"bad card '{text}'")
        rank, suit = t[:-1], t[-1]
        if rank == "1":
            rank = "10"
        if rank not in self.ranks or suit not in self.suits:
            raise ValueError(f"bad card '{text}'")
        return self.card_id(rank, suit)

    def parse_cards(self, text: str) -> list[int]:
        parts = [p for p in text.replace(",", " ").split() if p]
        return [self.parse_card(p) for p in parts]

    def beats(self, b: int, a: int, trump_suit: int) -> bool:
        """True if card b beats card a."""
        sb, sa = self.suit_of[b], self.suit_of[a]
        if sb == sa:
            return self.strength[b] > self.strength[a]
        return sb == trump_suit and self.raw["trump"]["trump_beats_any_non_trump"]
