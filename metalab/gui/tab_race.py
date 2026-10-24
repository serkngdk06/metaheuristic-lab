"""Tab 3 - PSO vs Genetic Algorithm, single race or N-run statistics.

Metaheuristics are stochastic: one run proves nothing. This tab repeats both
algorithms N times with different seeds and reports mean / std / best / worst,
a median convergence curve with an inter-quartile band, and a box plot.
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from ..algorithms import PSO, GeneticAlgorithm
from ..benchmarks import BENCHMARKS
from . import style as S
from .widgets import GA_SPEC, PSO_SPEC, SELECTION_OPTIONS, ParamForm, PlotPanel, error, scroll_column


class RaceTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.queue = queue.Queue()
        self.worker = None
        self._build()

    def _build(self):
        left = scroll_column(self)

        prob = ttk.LabelFrame(left, text="Problem", padding=(10, 6))
        prob.pack(fill="x")
        self.func_var = tk.StringVar(value="Rastrigin")
        cb = ttk.Combobox(prob, textvariable=self.func_var, state="readonly",
                          values=list(BENCHMARKS), width=28)
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", lambda e: self._check_dim())

        self.common = ParamForm(left, "Deney ayarları", [
            ("dim", "Boyut (değişken sayısı)", 10, int, (1, 100)),
            ("n", "Popülasyon / sürü boyutu", 30, int, (2, 500)),
            ("iters", "İterasyon sayısı", 200, int, (1, 5000)),
            ("runs", "Bağımsız koşu sayısı", 20, int, (1, 200)),
        ])
        self.common.pack(fill="x", pady=(8, 0))
        self.pso_form = ParamForm(left, "PSO", PSO_SPEC)
        self.pso_form.pack(fill="x", pady=(8, 0))
        self.ga_form = ParamForm(left, "Genetik Algoritma", GA_SPEC)
        self.ga_form.pack(fill="x", pady=(8, 0))
        self.sel_var = tk.StringVar(value="Turnuva (k=3)")
        ttk.Label(self.ga_form, text="Seçim yöntemi", style="Panel.TLabel").grid(
            row=len(GA_SPEC), column=0, sticky="w")
        ttk.Combobox(self.ga_form, textvariable=self.sel_var, state="readonly", width=14,
                     values=list(SELECTION_OPTIONS)).grid(row=len(GA_SPEC), column=1, pady=2)

        self.btn = ttk.Button(left, text="🏁  Yarıştır", style="Accent.TButton",
                              command=self.start)
        self.btn.pack(fill="x", pady=(10, 0))
        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.pack(fill="x", pady=(6, 0))
        self.status = ttk.Label(left, text="", style="Muted.TLabel")
        self.status.pack(anchor="w")
        ttk.Button(left, text="PNG kaydet",
                   command=lambda: self.plot.save_png("yaris.png")).pack(fill="x", pady=(6, 0))

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
        self.plot = PlotPanel(right, figsize=(9, 4.6))
        gs = self.plot.fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.25)
        self.ax = self.plot.fig.add_subplot(gs[0])
        self.ax_box = self.plot.fig.add_subplot(gs[1])
        self.plot.fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.12)

        cols = ("algo", "mean", "std", "best", "worst", "median", "wins", "time")
        heads = ("Algoritma", "Ortalama", "Std. sapma", "En iyi", "En kötü",
                 "Medyan", "Kazandığı koşu", "Süre / koşu")
        self.table = ttk.Treeview(right, columns=cols, show="headings", height=3)
        for c, h in zip(cols, heads):
            self.table.heading(c, text=h)
            self.table.column(c, anchor="center", width=105)
        self.verdict = ttk.Label(right, text="Parametreleri seçip 'Yarıştır'a basın.",
                                 font=S.FONT_BOLD)
        self.verdict.pack(side="bottom", anchor="w", pady=(6, 0))
        self.table.pack(side="bottom", fill="x", pady=(8, 0))
        self.plot.pack(fill="both", expand=True)

    def _check_dim(self):
        if not BENCHMARKS[self.func_var.get()].any_dim:
            self.common.set("dim", 2)

    # ------------------------------------------------------------------ #
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            c, p, g = self.common.get(), self.pso_form.get(), self.ga_form.get()
        except ValueError as exc:
            error(str(exc))
            return
        bench = BENCHMARKS[self.func_var.get()]
        if not bench.any_dim and c["dim"] != 2:
            error(f"{bench.name} yalnızca 2 boyutta tanımlı.")
            return
        sel = SELECTION_OPTIONS[self.sel_var.get()]
        self.btn.state(["disabled"])
        self.progress.config(maximum=c["runs"], value=0)
        self.worker = threading.Thread(target=self._work, args=(bench, c, p, g, sel), daemon=True)
        self.worker.start()
        self.after(100, self._poll)

    def _work(self, bench, c, p, g, sel):
        """Runs in a background thread so the window stays responsive."""
        res = {"PSO": {"curves": [], "final": [], "time": []},
               "GA": {"curves": [], "final": [], "time": []}}
        for run in range(c["runs"]):
            seed = 1000 + run
            algos = {
                "PSO": PSO(bench.func, c["dim"], bench.bounds, c["n"], c["iters"],
                           w_max=p["w_max"], w_min=p["w_min"], c1=p["c1"], c2=p["c2"],
                           v_max_ratio=p["vmax"], seed=seed),
                "GA": GeneticAlgorithm(bench.func, c["dim"], bench.bounds, c["n"], c["iters"],
                                       crossover_rate=g["pc"], mutation_rate=g["pm"],
                                       mutation_scale=g["sigma"], elitism=g["elitism"],
                                       selection=sel, seed=seed),
            }
            for name, algo in algos.items():
                t = time.perf_counter()
                algo.run()
                res[name]["time"].append(time.perf_counter() - t)
                res[name]["curves"].append(algo.history)
                res[name]["final"].append(algo.gbest_f)
            self.queue.put(("progress", run + 1))
        self.queue.put(("done", (bench, c, res)))

    def _poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "progress":
                    self.progress.config(value=payload)
                    self.status.config(text=f"Koşu {payload} tamamlandı")
                else:
                    self._show(*payload)
                    self.btn.state(["!disabled"])
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # ------------------------------------------------------------------ #
    def _show(self, bench, c, res):
        colors = {"PSO": S.PSO_COLOR, "GA": S.GA_COLOR}
        labels = {"PSO": "PSO", "GA": "Genetik Algoritma"}
        ax = self.ax
        ax.clear()
        eps = 1e-300
        for name, r in res.items():
            curves = np.maximum(np.array(r["curves"]), eps)
            med = np.median(curves, axis=0)
            q1, q3 = np.percentile(curves, [25, 75], axis=0)
            it = np.arange(len(med))
            ax.semilogy(it, med, color=colors[name], lw=2, label=f"{labels[name]} (medyan)")
            if c["runs"] > 1:
                ax.fill_between(it, q1, q3, color=colors[name], alpha=0.18)
        ax.set_title(f"{bench.name}  •  {c['dim']} boyut  •  {c['runs']} koşu"
                     + ("  (bant: %25–%75)" if c["runs"] > 1 else ""))
        ax.set_xlabel("iterasyon")
        ax.set_ylabel("en iyi uygunluk [log]")
        ax.legend()

        b = self.ax_box
        b.clear()
        data = [np.maximum(res[k]["final"], eps) for k in ("PSO", "GA")]
        bp = b.boxplot(data, patch_artist=True, widths=0.5)
        b.set_xticks([1, 2], ["PSO", "GA"])
        for patch, k in zip(bp["boxes"], ("PSO", "GA")):
            patch.set_facecolor(colors[k])
            patch.set_alpha(0.35)
        for med in bp["medians"]:
            med.set_color("black")
        for i, d in enumerate(data, start=1):
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(d))
            b.scatter(i + jitter, d, s=10, color="black", alpha=0.5, zorder=3)
        b.set_yscale("log")
        b.set_title("Son değerlerin dağılımı")
        self.plot.draw()

        pso_f, ga_f = np.array(res["PSO"]["final"]), np.array(res["GA"]["final"])
        wins = {"PSO": int(np.sum(pso_f < ga_f)), "GA": int(np.sum(ga_f < pso_f))}
        self.table.delete(*self.table.get_children())
        for k in ("PSO", "GA"):
            f = np.array(res[k]["final"])
            self.table.insert("", "end", values=(
                labels[k], f"{f.mean():.4e}", f"{f.std():.4e}", f"{f.min():.4e}",
                f"{f.max():.4e}", f"{np.median(f):.4e}", f"{wins[k]} / {c['runs']}",
                f"{1000 * np.mean(res[k]['time']):.0f} ms"))
        better = "PSO" if np.median(pso_f) < np.median(ga_f) else "Genetik Algoritma"
        self.verdict.config(text=f"Sonuç: {bench.name} ({c['dim']}B) probleminde medyan "
                                 f"değere göre daha başarılı olan → {better}")
        self.status.config(text="Tamamlandı")
