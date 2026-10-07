"""Manual correction of the tracked round (used when recognition is unsure)."""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QSpinBox,
                               QVBoxLayout)

from ..engine.core import ME, OPP
from ..rules import Rules
from ..tracker.state import RoundState

HELP = ("Cards are typed like  A♦ 10C KH 7s TD  (rank + suit letter D/C/H/S or symbol), separated by spaces.\n"
        "Only fill in what you know. Unknown cards in a pile go into the 'unknown' counters.")


class CorrectionDialog(QDialog):
    def __init__(self, rules: Rules, rs: RoundState | None, screen_hand: list[int], parent=None):
        super().__init__(parent)
        self.rules = rules
        self.setWindowTitle("Correct game state")
        self.result_state: RoundState | None = None
        R = rules
        if rs is None:
            rs = RoundState(rules, number=1)
            rs.my_hand = list(screen_hand)
            rs.opp_hand_size = rules.hand_size
            rs.deck_size = rules.n_cards - 2 * rules.hand_size
        self.base = rs
        names = lambda cs: " ".join(R.card_name(c) for c in cs)  # noqa: E731

        lay = QVBoxLayout(self)
        tip = QLabel(HELP)
        tip.setWordWrap(True)
        lay.addWidget(tip)
        form = QFormLayout()
        self.trump = QLineEdit(R.card_name(rs.trump_card) if rs.trump_card is not None else "")
        self.trump_in_deck = QComboBox()
        self.trump_in_deck.addItems(["under the deck", "taken by me", "taken by opponent"])
        if not rs.trump_in_deck:
            self.trump_in_deck.setCurrentIndex(1 if rs.trump_card in rs.my_hand else 2)
        self.hand = QLineEdit(names(rs.my_hand or screen_hand))
        self.my_pile = QLineEdit(names(rs.my_pile))
        self.my_unknown = QSpinBox()
        self.my_unknown.setRange(0, rules.n_cards)
        self.my_unknown.setValue(rs.my_pile_unknown)
        self.opp_pile = QLineEdit(names(rs.opp_pile))
        self.opp_unknown = QSpinBox()
        self.opp_unknown.setRange(0, rules.n_cards)
        self.opp_unknown.setValue(rs.opp_pile_unknown)
        self.opp_known = QLineEdit(names([c for c in rs.opp_known if c != rs.trump_card]))
        self.opp_hand = QSpinBox()
        self.opp_hand.setRange(0, rules.hand_size)
        self.opp_hand.setValue(rs.opp_hand_size)
        self.deck = QSpinBox()
        self.deck.setRange(0, rules.n_cards)
        self.deck.setValue(rs.deck_size)
        self.my_tricks = QSpinBox()
        self.my_tricks.setRange(0, 40)
        self.my_tricks.setValue(rs.my_tricks)
        self.opp_tricks = QSpinBox()
        self.opp_tricks.setRange(0, 40)
        self.opp_tricks.setValue(rs.opp_tricks)
        self.leader = QComboBox()
        self.leader.addItems(["me", "opponent"])
        self.leader.setCurrentIndex(0 if rs.next_leader in (None, ME) else 1)
        self.stake = QComboBox()
        for lv in rules.stake_levels:
            self.stake.addItem(f"{lv.get('name', lv['value'])} ({lv['value']})", lv["value"])
        idx = max(0, self.stake.findData(rs.stake))
        self.stake.setCurrentIndex(idx)

        form.addRow("Trump card", self.trump)
        form.addRow("Trump card is", self.trump_in_deck)
        form.addRow("My hand", self.hand)
        form.addRow("My pile (known)", self.my_pile)
        form.addRow("My pile unknown cards", self.my_unknown)
        form.addRow("Opponent pile (known)", self.opp_pile)
        form.addRow("Opponent pile unknown cards", self.opp_unknown)
        form.addRow("Known cards in opponent hand", self.opp_known)
        form.addRow("Opponent hand size", self.opp_hand)
        form.addRow("Cards left in deck (incl. trump)", self.deck)
        form.addRow("Tricks won by me", self.my_tricks)
        form.addRow("Tricks won by opponent", self.opp_tricks)
        form.addRow("Next to lead", self.leader)
        form.addRow("Stake", self.stake)
        lay.addLayout(form)
        self.err = QLabel("")
        self.err.setStyleSheet("color:#d1242f")
        self.err.setWordWrap(True)
        lay.addWidget(self.err)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _accept(self):
        R = self.rules
        try:
            rs = RoundState(R, number=self.base.number or 1)
            trump = R.parse_cards(self.trump.text())
            if len(trump) != 1:
                raise ValueError("enter exactly one trump card")
            rs.trump_card = trump[0]
            rs.trump_suit = R.suit_of[trump[0]]
            rs.my_hand = R.parse_cards(self.hand.text())
            rs.my_pile = R.parse_cards(self.my_pile.text())
            rs.my_pile_unknown = self.my_unknown.value()
            rs.opp_pile = R.parse_cards(self.opp_pile.text())
            rs.opp_pile_unknown = self.opp_unknown.value()
            rs.opp_known = R.parse_cards(self.opp_known.text())
            rs.opp_hand_size = self.opp_hand.value()
            rs.deck_size = self.deck.value()
            where = self.trump_in_deck.currentIndex()
            rs.trump_in_deck = where == 0 and rs.deck_size > 0
            if where == 2 and rs.trump_card not in rs.opp_known and rs.trump_card not in rs.opp_pile \
                    and rs.trump_card not in rs.my_pile:
                rs.opp_known.append(rs.trump_card)
            rs.my_tricks = self.my_tricks.value()
            rs.opp_tricks = self.opp_tricks.value()
            rs.next_leader = ME if self.leader.currentIndex() == 0 else OPP
            rs.stake = int(self.stake.currentData())
            errs = rs.consistency_errors()
            if errs:
                raise ValueError("; ".join(errs))
        except ValueError as exc:
            self.err.setText(str(exc))
            return
        self.result_state = rs
        self.accept()
