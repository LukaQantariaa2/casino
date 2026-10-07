"""Small Win32 helpers (no-ops on other platforms). Nothing here sends input to other windows."""
from __future__ import annotations

import ctypes
import sys
import threading

IS_WIN = sys.platform == "win32"

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WDA_EXCLUDEFROMCAPTURE = 0x00000011
WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000

VK = {f"F{i}": 0x6F + i for i in range(1, 13)}   # F1 = 0x70


def set_dpi_aware():
    if not IS_WIN:
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def exclude_from_capture(hwnd: int) -> bool:
    """Hide our overlay from screen capture (Windows 10 2004+), so it never covers what we read."""
    if not IS_WIN:
        return False
    try:
        return bool(ctypes.windll.user32.SetWindowDisplayAffinity(ctypes.c_void_p(hwnd), WDA_EXCLUDEFROMCAPTURE))
    except Exception:
        return False


def set_click_through(hwnd: int, enabled: bool):
    if not IS_WIN:
        return
    u = ctypes.windll.user32
    get = getattr(u, "GetWindowLongPtrW", u.GetWindowLongW)
    put = getattr(u, "SetWindowLongPtrW", u.SetWindowLongW)
    get.restype = ctypes.c_ssize_t
    ex = get(ctypes.c_void_p(hwnd), GWL_EXSTYLE)
    if enabled:
        ex |= WS_EX_LAYERED | WS_EX_TRANSPARENT
    else:
        ex &= ~WS_EX_TRANSPARENT
    put(ctypes.c_void_p(hwnd), GWL_EXSTYLE, ctypes.c_ssize_t(ex))


class HotkeyThread(threading.Thread):
    """Global hotkeys via RegisterHotKey; calls callback(name) from this thread."""

    def __init__(self, mapping: dict[str, str], callback):
        super().__init__(daemon=True)
        self.mapping = mapping            # action -> "F8"
        self.callback = callback
        self.registered: list[str] = []
        self.failed: list[str] = []

    def run(self):
        if not IS_WIN:
            return
        import ctypes.wintypes as wt
        u = ctypes.windll.user32
        ids = {}
        for i, (action, key) in enumerate(self.mapping.items(), start=1):
            vk = VK.get(str(key).upper())
            if vk is None:
                self.failed.append(f"{action}={key}")
                continue
            if u.RegisterHotKey(None, i, MOD_NOREPEAT, vk):
                ids[i] = action
                self.registered.append(f"{key}: {action}")
            else:
                self.failed.append(f"{action}={key}")
        msg = wt.MSG()
        while u.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == WM_HOTKEY and msg.wParam in ids:
                try:
                    self.callback(ids[msg.wParam])
                except Exception:
                    pass
