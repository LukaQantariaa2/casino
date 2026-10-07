"""Always-on-top recommendation overlay (PySide6).

The overlay only displays information. It never sends clicks or keys to the
browser; you play every move yourself.
"""
from __future__ import annotations

import html

from PySide6.QtCore import QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from ..controller import Snapshot
from ..engine.advisor import Recommendation
from . import winapi

STATUS_COLORS = {"OK": "#2ecc71", "UNCERTAIN": "#f1c40f", "PAUSED": "#e74c3c", "WAITING": "#95a5a6"}
CELL_STYLE = {
    "mine": ("#1f6feb", "#ffffff"),
    "table": ("#8957e5", "#ffffff"),
    "my_pile": ("#1b3a24", "#7ee2a8"),
    "opp_pile": ("#3a1b1b", "#f19999"),
    "opp_hand": ("#b62324", "#ffffff"),
    "trump": ("#9e6a03", "#ffffff"),
    "unseen": ("#2d333b", "#e6edf3"),
}
PHASE_TEXT = {"lead": "YOUR LEAD", "respond": "YOUR ANSWER", "claim": "YOU WON THE TRICK",
              "raise_offer": "OPPONENT RAISES", "wait": "waiting", "round_over": "round over"}


def card_html(text: str) -> str:
    """Colour suit symbols for a dark background."""
    out = html.escape(text)
    for sym, col in (("♥", "#ff6b6b"), ("♦", "#ff6b9a"), ("♣", "#9fd3ff"), ("♠", "#e6edf3")):
        out = out.replace(sym, f'<span style="color:{col}">{sym}</span>')
    return out


class Bridge(QObject):
    snapshot = Signal(object)
    recommendation = Signal(object, object)
    hotkey = Signal(str)


class Overlay(QWidget):
    def __init__(self, controller, cfg: dict, rules):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.ctrl = controller
        self.cfg = cfg
        self.rules = rules
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("Bura Assistant")
        oc = cfg.get("overlay", {})
        self.setWindowOpacity(float(oc.get("opacity", 0.92)))
        self.click_through = bool(oc.get("click_through", False))
        self._drag: QPoint | None = None
        self._snap: Snapshot | None = None
        self._rec: Recommendation | None = None
        self._rec_key = None
        fs = int(oc.get("font_size", 11))

        self.frame = QFrame(self)
        self.frame.setObjectName("panel")
        self.frame.setStyleSheet(
            "#panel { background: rgba(13,17,23,235); border: 1px solid #30363d; border-radius: 10px; }"
            "QLabel { color: #e6edf3; }"
            "QPushButton { background:#21262d; color:#e6edf3; border:1px solid #30363d; border-radius:5px;"
            " padding:2px 7px; }"
            "QPushButton:hover { background:#30363d; }")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.frame)
        lay = QVBoxLayout(self.frame)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        # header
        head = QHBoxLayout()
        self.status_lbl = QLabel("●  WAITING")
        self.status_lbl.setFont(QFont("Segoe UI", fs, QFont.Weight.Bold))
        self.phase_lbl = QLabel("")
        self.phase_lbl.setFont(QFont("Segoe UI", fs))
        self.phase_lbl.setStyleSheet("color:#8b949e")
        head.addWidget(self.status_lbl)
        head.addStretch(1)
        head.addWidget(self.phase_lbl)
        lay.addLayout(head)

        # recommendation
        self.rec_lbl = QLabel("—")
        self.rec_lbl.setFont(QFont("Segoe UI", fs + 5, QFont.Weight.Bold))
        self.rec_lbl.setWordWrap(True)
        self.rec_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.detail_lbl = QLabel("")
        self.detail_lbl.setWordWrap(True)
        self.detail_lbl.setStyleSheet("color:#8b949e")
        self.extra_lbl = QLabel("")
        self.extra_lbl.setWordWrap(True)
        self.extra_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.alt_lbl = QLabel("")
        self.alt_lbl.setWordWrap(True)
        self.alt_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.alt_lbl.setStyleSheet("color:#adbac7")
        for w in (self.rec_lbl, self.detail_lbl, self.extra_lbl, self.alt_lbl):
            lay.addWidget(w)

        # warning
        self.warn_lbl = QLabel("")
        self.warn_lbl.setWordWrap(True)
        self.warn_lbl.setStyleSheet("background:#3d1d1d; color:#ffb4b4; border-radius:6px; padding:6px;")
        self.warn_lbl.hide()
        lay.addWidget(self.warn_lbl)

        # score / round info
        self.info_lbl = QLabel("")
        self.info_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.info_lbl.setWordWrap(True)
        lay.addWidget(self.info_lbl)

        # card grid
        self.grid_w = QWidget()
        self.grid = QGridLayout(self.grid_w)
        self.grid.setSpacing(2)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.cells: list[list[QLabel]] = []
        for r in range(len(rules.suits)):
            row = []
            for c in range(len(rules.ranks)):
                lbl = QLabel("")
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setFixedHeight(18)
                lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                self.grid.addWidget(lbl, r, c)
                row.append(lbl)
            self.cells.append(row)
        if oc.get("show_unseen_cards", True):
            lay.addWidget(self.grid_w)
            legend = QLabel('<span style="color:#1f6feb">■</span> mine  '
                            '<span style="color:#b62324">■</span> opp holds  '
                            '<span style="color:#2d333b">■</span> unseen  '
                            '<span style="color:#7ee2a8">■</span> my pile  '
                            '<span style="color:#f19999">■</span> opp pile')
            legend.setTextFormat(Qt.TextFormat.RichText)
            legend.setStyleSheet("color:#8b949e; font-size:9px")
            lay.addWidget(legend)
        else:
            self.grid_w.hide()

        # history
        self.hist_lbl = QLabel("")
        self.hist_lbl.setWordWrap(True)
        self.hist_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.hist_lbl.setStyleSheet("color:#8b949e; font-size:10px")
        lay.addWidget(self.hist_lbl)

        # buttons (two rows so the panel stays narrow)
        btn_rows = QVBoxLayout()
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        btn_rows.addLayout(row1)
        btn_rows.addLayout(row2)
        btns = row1
        hk = cfg.get("hotkeys", {})

        def mk(text, fn, tip=""):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            btns.addWidget(b)
            return b

        self.hide_btn = mk(f"Hide tips {hk.get('pause_resume', 'F8')}", self.ctrl.toggle_recommendations)
        mk(f"Correct {hk.get('correction', 'F9')}", self.open_correction)
        mk(f"Reset {hk.get('reset_round', 'F10')}", self.ctrl.reset_round)
        btns = row2
        mk("Stake −", lambda: self.ctrl.adjust_stake(-1), "lower the tracked stake by one level")
        mk("Stake +", lambda: self.ctrl.adjust_stake(+1), "raise the tracked stake by one level")
        mk("Quit", self.close_app, "quit")
        lay.addLayout(btn_rows)
        self.src_lbl = QLabel("")
        self.src_lbl.setStyleSheet("color:#6e7681; font-size:9px")
        lay.addWidget(self.src_lbl)

        self.setFixedWidth(int(oc.get("width", 305)))
        self.move(int(oc.get("x", 290)), int(oc.get("y", 255)))

        # thread bridge
        self.bridge = Bridge()
        self.bridge.snapshot.connect(self._on_snapshot)
        self.bridge.recommendation.connect(self._on_rec)
        self.bridge.hotkey.connect(self._on_hotkey)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(500)

    # ---------------------------------------------------------------- window plumbing
    def showEvent(self, ev):
        super().showEvent(ev)
        hwnd = int(self.winId())
        if self.cfg.get("overlay", {}).get("exclude_from_capture", True):
            winapi.exclude_from_capture(hwnd)
        winapi.set_click_through(hwnd, self.click_through)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag = ev.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, ev):
        if self._drag is not None:
            self.move(ev.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def toggle_click_through(self):
        self.click_through = not self.click_through
        winapi.set_click_through(int(self.winId()), self.click_through)
        self._render()

    def close_app(self):
        self.ctrl.stop()
        QApplication.quit()

    # ---------------------------------------------------------------- data in (any thread)
    def push_snapshot(self, snap):
        self.bridge.snapshot.emit(snap)

    def push_recommendation(self, rec, key):
        self.bridge.recommendation.emit(rec, key)

    def push_hotkey(self, name):
        self.bridge.hotkey.emit(name)

    def _on_snapshot(self, snap: Snapshot):
        self._snap = snap
        self._render()

    def _on_rec(self, rec, key):
        self._rec = rec
        self._rec_key = key
        self._render()

    def _on_hotkey(self, name: str):
        if name == "pause_resume":
            self.ctrl.toggle_recommendations()
        elif name == "correction":
            self.open_correction()
        elif name == "reset_round":
            self.ctrl.reset_round()
        elif name == "toggle_click_through":
            self.toggle_click_through()

    def _tick(self):
        if self._snap is not None:
            self.src_lbl.setText(f"{self._snap.source}   ·   {self._snap.fps:.1f} fps   ·   {self._snap.ocr_status}"
                                 f"{'   ·   click-through ON (F7)' if self.click_through else ''}")

    # ---------------------------------------------------------------- correction dialog
    def open_correction(self):
        from .correction import CorrectionDialog
        rs = self.ctrl.current_round_copy()
        dlg = CorrectionDialog(self.rules, rs, self.ctrl.last_hand(), self)
        if dlg.exec():
            try:
                self.ctrl.apply_correction(dlg.result_state)
            except Exception as exc:
                self.warn_lbl.setText(f"Correction rejected: {exc}")
                self.warn_lbl.show()

    @staticmethod
    def _match_text(s) -> str:
        m = s.match or {}
        if m.get("my_score") is None and m.get("opp_score") is None:
            return ""
        q = lambda v: "?" if v is None else v  # noqa: E731
        return (f"<br>match: me {q(m.get('my_score'))} – {q(m.get('opp_score'))} opp · "
                f"target {q(m.get('target'))} · bet {q(m.get('bet'))}")

    # ---------------------------------------------------------------- render
    def _render(self):
        s = self._snap
        if s is None:
            return
        col = STATUS_COLORS.get(s.status, "#e6edf3")
        self.status_lbl.setText(f'<span style="color:{col}">●</span>  {s.status}')
        stake = f"stake {html.escape(s.stake_name)}" + (f" ({s.stake})" if s.stake_name != str(s.stake) else "")
        self.phase_lbl.setText(PHASE_TEXT.get(s.phase, s.phase))

        rec = self._rec
        show_rec = (rec is not None and s.status == "OK" and not s.hidden
                    and s.phase in ("lead", "respond", "claim", "raise_offer"))
        if s.hidden:
            self.rec_lbl.setText('<span style="color:#6e7681">tips hidden (F8)</span>')
            self.detail_lbl.setText("")
            self.extra_lbl.setText("")
            self.alt_lbl.setText("")
        elif show_rec:
            self.rec_lbl.setText(card_html(rec.headline))
            self.detail_lbl.setText(rec.detail)
            self.extra_lbl.setText("<br>".join(f'<span style="color:#f0b72f">▲ {card_html(e)}</span>'
                                               for e in rec.extra))
            self.alt_lbl.setText("<br>".join(f"· {card_html(a)}" for a in rec.alternatives))
        else:
            if s.status == "OK" and s.phase in ("lead", "respond", "claim", "raise_offer"):
                txt = '<span style="color:#8b949e">calculating…</span>'
            elif s.status == "OK":
                txt = '<span style="color:#8b949e">waiting for your turn</span>'
            else:
                txt = '<span style="color:#8b949e">no recommendation</span>'
            self.rec_lbl.setText(txt)
            self.detail_lbl.setText("")
            self.extra_lbl.setText("")
            self.alt_lbl.setText("")

        if s.status in ("PAUSED", "UNCERTAIN") and s.reasons:
            prefix = "Recommendations paused — " if s.status == "PAUSED" else "Checking — "
            tail = "  Press F9 to correct the state, or F10 to reset and wait for the next deal." \
                if s.status == "PAUSED" else ""
            self.warn_lbl.setText(prefix + "; ".join(s.reasons) + tail)
            self.warn_lbl.show()
        elif s.status == "WAITING" and s.reasons:
            self.warn_lbl.setText("; ".join(s.reasons))
            self.warn_lbl.setStyleSheet("background:#1c2128; color:#adbac7; border-radius:6px; padding:6px;")
            self.warn_lbl.show()
        else:
            self.warn_lbl.setStyleSheet("background:#3d1d1d; color:#ffb4b4; border-radius:6px; padding:6px;")
            self.warn_lbl.hide()

        if s.round_no:
            mu = f" + {s.my_unknown}?" if s.my_unknown else ""
            ou = f" + {s.opp_unknown}?" if s.opp_unknown else ""
            self.info_lbl.setText(
                f"<b>Me {s.my_points}{mu}</b> &nbsp;vs&nbsp; <b>Opp {s.opp_points}{ou}</b> &nbsp;(need {s.win_points})<br>"
                f"trump {card_html(s.trump)} · deck {s.deck} · opp holds {s.opp_hand}<br>"
                f"round {s.round_no} · {stake}{self._match_text(s)}")
        else:
            self.info_lbl.setText("")

        for r, row in enumerate(s.card_grid):
            for c, (label, st) in enumerate(row):
                if r < len(self.cells) and c < len(self.cells[r]):
                    bg, fg = CELL_STYLE.get(st, ("#2d333b", "#e6edf3"))
                    cell = self.cells[r][c]
                    cell.setText(label)
                    cell.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:3px; font-size:10px;")
        if not s.card_grid:
            for row in self.cells:
                for cell in row:
                    cell.setText("")
                    cell.setStyleSheet("background:#161b22; border-radius:3px;")

        self.hist_lbl.setText("<br>".join(card_html(h) for h in s.history[-6:]))
        self.adjustSize()
