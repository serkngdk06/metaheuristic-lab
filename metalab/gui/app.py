"""Main window: header + four tabs."""

import tkinter as tk
from tkinter import ttk

from .. import __version__
from . import style as S
from .tab_neural import NeuralTab
from .tab_race import RaceTab
from .tab_teach import TeachTab
from .tab_visual import VisualTab


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MetaLab — Metasezgisel Optimizasyon Laboratuvarı")
        self.geometry("1440x880")
        self.minsize(1000, 640)
        S.apply_theme(self)

        header = tk.Frame(self, bg=S.HEADER, height=56)
        header.pack(fill="x")
        tk.Label(header, text="MetaLab", bg=S.HEADER, fg="white",
                 font=S.FONT_TITLE).pack(side="left", padx=(16, 8), pady=10)
        tk.Label(header, text="Parçacık Sürü Optimizasyonu  •  Genetik Algoritma  •  "
                              "PSO ile Yapay Sinir Ağı Eğitimi",
                 bg=S.HEADER, fg="#9ca3af", font=S.FONT).pack(side="left", pady=10)
        tk.Label(header, text=f"v{__version__}", bg=S.HEADER, fg="#6b7280",
                 font=S.FONT).pack(side="right", padx=16)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.tabs = {}
        for key, cls, label in [
            ("visual", VisualTab, "  Sürü Görselleştirme  "),
            ("teach", TeachTab, "  Eğitim Modu (Adım Adım)  "),
            ("race", RaceTab, "  Algoritma Yarışı  "),
            ("neural", NeuralTab, "  PSO ile Sinir Ağı  "),
        ]:
            tab = cls(nb)
            nb.add(tab, text=label)
            self.tabs[key] = tab
        self.notebook = nb
        self.protocol("WM_DELETE_WINDOW", self._quit)

    def _quit(self):
        self.quit()
        self.destroy()


def main():
    App().mainloop()
