"""Tab 1 - live animation of the swarm (or GA population) on a test function."""

import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from ..algorithms import PSO, GeneticAlgorithm
from ..benchmarks import BENCHMARKS
from ..expression import ExpressionError, parse_expression
from . import style as S
from .widgets import GA_SPEC, PSO_SPEC, SELECTION_OPTIONS, ParamForm, PlotPanel, error, stat_row, scroll_column

TEACHER = "Ders örneği: |x² + 2x − 3|"
CUSTOM = "Özel fonksiyon (aşağıya yazın)"


def resolve_function(choice, custom_text, low, high):
    """Return (func, dim, bounds, optimum) for the selected item."""
    if choice in BENCHMARKS:
        b = BENCHMARKS[choice]
        return b.func, 2, (low, high), b.optimum
    text = "abs(x**2 + 2*x - 3)" if choice == TEACHER else custom_text
    func, dim = parse_expression(text)
    return func, dim, (low, high), None


class VisualTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.opt = None
        self.running = False
        self._job = None
        self._build_controls()
        self._build_plots()
        self._on_function_change()

    # ------------------------------------------------------------ UI --- #
    def _build_controls(self):
        left = scroll_column(self)

        box = ttk.LabelFrame(left, text="Problem", padding=(10, 6))
        box.pack(fill="x")
        self.func_var = tk.StringVar(value="Rastrigin")
        cb = ttk.Combobox(box, textvariable=self.func_var, state="readonly", width=30,
                          values=list(BENCHMARKS) + [TEACHER, CUSTOM])
        cb.grid(row=0, column=0, columnspan=4, sticky="ew")
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_function_change())
        self.desc = ttk.Label(box, text="", style="Muted.TLabel", wraplength=240)
        self.desc.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 2))
        ttk.Label(box, text="f(x, y) =", style="Panel.TLabel").grid(row=2, column=0, sticky="w")
        self.expr_var = tk.StringVar(value="sin(x)*cos(y) + 0.1*(x**2 + y**2)")
        self.expr_entry = ttk.Entry(box, textvariable=self.expr_var, width=24)
        self.expr_entry.grid(row=2, column=1, columnspan=3, sticky="ew", pady=2)
        ttk.Label(box, text="Alt sınır", style="Panel.TLabel").grid(row=3, column=0, sticky="w")
        self.low_var = tk.StringVar()
        ttk.Entry(box, textvariable=self.low_var, width=7).grid(row=3, column=1, pady=2)
        ttk.Label(box, text="Üst", style="Panel.TLabel").grid(row=3, column=2)
        self.high_var = tk.StringVar()
        ttk.Entry(box, textvariable=self.high_var, width=7).grid(row=3, column=3, pady=2)

        algo = ttk.LabelFrame(left, text="Algoritma", padding=(10, 6))
        algo.pack(fill="x", pady=(8, 0))
        self.algo_var = tk.StringVar(value="PSO")
        for i, name in enumerate(["PSO", "Genetik Algoritma"]):
            ttk.Radiobutton(algo, text=name, value=name, variable=self.algo_var,
                            command=self._on_algo_change).grid(row=0, column=i, sticky="w", padx=4)

        self.common = ParamForm(left, "Genel", [
            ("n", "Parçacık / birey sayısı", 30, int, (2, 500)),
            ("iters", "İterasyon sayısı", 100, int, (1, 5000)),
            ("seed", "Seed (boş = rastgele)", None, "seed", None),
        ])
        self.common.pack(fill="x", pady=(8, 0))

        self.algo_holder = ttk.Frame(left)
        self.algo_holder.pack(fill="x", pady=(8, 0))
        self.pso_form = ParamForm(self.algo_holder, "PSO parametreleri", PSO_SPEC)
        self.ga_form = ParamForm(self.algo_holder, "GA parametreleri", GA_SPEC)
        self.sel_var = tk.StringVar(value="Turnuva (k=3)")
        ttk.Label(self.ga_form, text="Seçim yöntemi", style="Panel.TLabel").grid(
            row=len(GA_SPEC), column=0, sticky="w")
        ttk.Combobox(self.ga_form, textvariable=self.sel_var, state="readonly", width=14,
                     values=list(SELECTION_OPTIONS)).grid(row=len(GA_SPEC), column=1, pady=2)
        self.pso_form.pack(fill="x")

        sp = ttk.LabelFrame(left, text="Animasyon", padding=(10, 6))
        sp.pack(fill="x", pady=(8, 0))
        ttk.Label(sp, text="Yavaş", style="Muted.TLabel").pack(side="left")
        self.speed = tk.IntVar(value=70)
        ttk.Scale(sp, from_=1, to=100, variable=self.speed, orient="horizontal").pack(
            side="left", fill="x", expand=True, padx=4)
        ttk.Label(sp, text="Hızlı", style="Muted.TLabel").pack(side="left")
        self.show_vel = tk.BooleanVar(value=True)

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(10, 0))
        self.start_btn = ttk.Button(btns, text="▶  Başlat", style="Accent.TButton",
                                    command=self.start)
        self.start_btn.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.pause_btn = ttk.Button(btns, text="⏸ Duraklat", command=self.toggle_pause)
        self.pause_btn.grid(row=1, column=0, sticky="ew", pady=(4, 0), padx=(0, 2))
        ttk.Button(btns, text="⏭ Tek adım", command=self.single_step).grid(
            row=1, column=1, sticky="ew", pady=(4, 0), padx=(2, 0))
        ttk.Button(btns, text="PNG kaydet", command=lambda: self.plot.save_png("pso.png")).grid(
            row=2, column=0, sticky="ew", pady=(4, 0), padx=(0, 2))
        ttk.Button(btns, text="CSV dışa aktar", command=self.export_csv).grid(
            row=2, column=1, sticky="ew", pady=(4, 0), padx=(2, 0))
        btns.columnconfigure((0, 1), weight=1)

        res = ttk.LabelFrame(left, text="Sonuçlar", padding=(10, 6))
        res.pack(fill="x", pady=(8, 0))
        self.v_iter = stat_row(res, 0, "İterasyon")
        self.v_best = stat_row(res, 1, "En iyi uygunluk")
        self.v_pos = stat_row(res, 2, "En iyi konum")
        self.v_evals = stat_row(res, 3, "Fonksiyon çağrısı")
        res.columnconfigure(1, weight=1)

    def _build_plots(self):
        self.plot = PlotPanel(self, figsize=(9, 6.5))
        self.plot.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
        gs = self.plot.fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.28)
        self.ax = self.plot.fig.add_subplot(gs[0])
        self.ax_conv = self.plot.fig.add_subplot(gs[1])
        self.plot.fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.1)

    # ------------------------------------------------------- events --- #
    def _on_function_change(self):
        choice = self.func_var.get()
        if choice in BENCHMARKS:
            b = BENCHMARKS[choice]
            self.desc.config(text=b.description)
            self.low_var.set(b.bounds[0])
            self.high_var.set(b.bounds[1])
            self.expr_entry.state(["disabled"])
        elif choice == TEACHER:
            self.desc.config(text="Hocanın psoVisual.py örneği: f(x)=0 yapan x aranır. "
                                  "Kökler x = 1 ve x = −3.")
            self.low_var.set(-6)
            self.high_var.set(4)
            self.expr_entry.state(["disabled"])
        else:
            self.desc.config(text="x (ve isterseniz y) kullanın. Örn: x**2 + 2*x - 3, "
                                  "sin(x)*y, sqrt(x**2+y**2). Fonksiyon küçültülür.")
            self.expr_entry.state(["!disabled"])
        self._preview()

    def _on_algo_change(self):
        self.pso_form.pack_forget()
        self.ga_form.pack_forget()
        (self.pso_form if self.algo_var.get() == "PSO" else self.ga_form).pack(fill="x")

    def _read_problem(self):
        try:
            low = float(self.low_var.get().replace(",", "."))
            high = float(self.high_var.get().replace(",", "."))
        except ValueError:
            raise ValueError("Sınırlar sayı olmalı.") from None
        if low >= high:
            raise ValueError("Alt sınır, üst sınırdan küçük olmalı.")
        try:
            return resolve_function(self.func_var.get(), self.expr_var.get(), low, high)
        except ExpressionError as exc:
            raise ValueError(str(exc)) from None

    def _preview(self):
        """Draw the landscape without particles."""
        try:
            self.func, self.dim, self.bounds, _ = self._read_problem()
        except ValueError:
            return
        self._draw_background()
        self.ax_conv.clear()
        self.ax_conv.set_title("Yakınsama eğrisi")
        self.plot.draw()

    # ------------------------------------------------------ drawing --- #
    def _draw_background(self):
        ax = self.ax
        ax.clear()
        low, high = self.bounds
        if self.dim == 1:
            xs = np.linspace(low, high, 600)
            ax.plot(xs, self.func(xs[:, None]), color="#334e68", lw=2)
            ax.set_xlabel("x")
            ax.set_ylabel("uygunluk f(x)")
            self.star = ax.scatter([], [], s=320, marker="*", c="#facc15",
                                   edgecolors="black", zorder=5, label="gbest")
            self.scat = ax.scatter([], [], s=55, c=S.PSO_COLOR, edgecolors="white",
                                   zorder=6, label="parçacıklar")
            self.quiv = None
            ax.legend(loc="upper center")
        else:
            g = np.linspace(low, high, 220)
            X, Y = np.meshgrid(g, g)
            Z = self.func(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
            Zs = np.log1p(Z - Z.min())  # log scale -> structure is visible
            ax.contourf(X, Y, Zs, levels=40, cmap="viridis")
            ax.contour(X, Y, Zs, levels=15, colors="white", linewidths=0.3, alpha=0.5)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            self.quiv = None
            self.scat = ax.scatter([], [], s=34, c="white", edgecolors="black",
                                   linewidths=0.8, zorder=5)
            self.trail, = ax.plot([], [], color="#f87171", lw=1.2, alpha=0.9, zorder=4)
            self.star = ax.scatter([], [], s=320, marker="*", c="#facc15",
                                   edgecolors="black", zorder=6)
        ax.set_title(self.func_var.get())

    def _update_plot(self, details=None):
        o = self.opt
        P = o.positions
        if self.dim == 1:
            self.scat.set_offsets(np.column_stack([P[:, 0], o.func(P)]))
            self.star.set_offsets([[o.gbest_x[0], o.gbest_f]])
        else:
            self.scat.set_offsets(P)
            self.star.set_offsets([o.gbest_x])
            self.gbest_path.append(o.gbest_x.copy())
            path = np.array(self.gbest_path)
            self.trail.set_data(path[:, 0], path[:, 1])
            if self.quiv is not None:
                self.quiv.remove()
                self.quiv = None
            if isinstance(o, PSO) and details is not None:
                # arrows: where each particle came from and the velocity it used
                xo, vn = details["x_old"], details["v_new"]
                self.quiv = self.ax.quiver(xo[:, 0], xo[:, 1], vn[:, 0], vn[:, 1],
                                           color="white", angles="xy", scale_units="xy",
                                           scale=1, width=0.004, alpha=0.75, zorder=4)
        self.ax.set_title(f"{self.func_var.get()}  •  iterasyon {o.iteration}/{o.max_iter}")

        a = self.ax_conv
        a.clear()
        hist = np.array(o.history)
        color = S.PSO_COLOR if isinstance(o, PSO) else S.GA_COLOR
        positive = np.all(hist > 0)
        plot = a.semilogy if positive else a.plot
        plot(hist, color=color, lw=2, label="en iyi (gbest)")
        mean = np.array(o.mean_history)
        if positive and np.all(mean > 0):
            plot(mean, color=color, lw=1, alpha=0.4, ls="--", label="sürü ortalaması")
        a.set_xlim(0, o.max_iter)
        a.set_title("Yakınsama eğrisi")
        a.set_xlabel("iterasyon")
        a.set_ylabel("uygunluk (küçük = iyi)" + ("  [log]" if positive else ""))
        a.legend(loc="upper right")

        self.v_iter.set(f"{o.iteration} / {o.max_iter}")
        self.v_best.set(f"{o.gbest_f:.6g}")
        self.v_pos.set("(" + ", ".join(f"{v:.4f}" for v in o.gbest_x) + ")")
        self.v_evals.set(str(o.evaluations))
        self.plot.draw()

    # ------------------------------------------------------ control --- #
    def _create_optimizer(self):
        self.func, self.dim, self.bounds, _ = self._read_problem()
        c = self.common.get()
        if self.algo_var.get() == "PSO":
            p = self.pso_form.get()
            return PSO(self.func, self.dim, self.bounds, c["n"], c["iters"],
                       w_max=p["w_max"], w_min=p["w_min"], c1=p["c1"], c2=p["c2"],
                       v_max_ratio=p["vmax"], seed=c["seed"])
        g = self.ga_form.get()
        return GeneticAlgorithm(self.func, self.dim, self.bounds, c["n"], c["iters"],
                                crossover_rate=g["pc"], mutation_rate=g["pm"],
                                mutation_scale=g["sigma"], elitism=g["elitism"],
                                selection=SELECTION_OPTIONS[self.sel_var.get()],
                                seed=c["seed"])

    def start(self):
        self._cancel()
        try:
            self.opt = self._create_optimizer()
        except ValueError as exc:
            error(str(exc))
            return
        self._draw_background()
        self.gbest_path = []
        self._update_plot()
        self.running = True
        self.pause_btn.config(text="⏸ Duraklat")
        self._job = self.after(300, self._tick)

    def _tick(self):
        if not self.running or self.opt is None or self.opt.done:
            self.running = False
            return
        details = self.opt.step()
        self._update_plot(details)
        delay = int(np.interp(self.speed.get(), [1, 100], [600, 10]))
        self._job = self.after(delay, self._tick)

    def toggle_pause(self):
        if self.opt is None or self.opt.done:
            return
        self.running = not self.running
        self.pause_btn.config(text="⏸ Duraklat" if self.running else "▶ Devam")
        if self.running:
            self._tick()

    def single_step(self):
        if self.opt is None:
            self.start()
            self.running = False
            self._cancel()
            return
        self.running = False
        self.pause_btn.config(text="▶ Devam")
        if not self.opt.done:
            self._update_plot(self.opt.step())

    def _cancel(self):
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None

    def export_csv(self):
        if self.opt is None:
            messagebox.showinfo("Bilgi", "Önce optimizasyonu çalıştırın.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            initialfile="yakinsama.csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iterasyon", "en_iyi_uygunluk", "ortalama_uygunluk"])
            for i, (b, m) in enumerate(zip(self.opt.history, self.opt.mean_history)):
                w.writerow([i, b, m])
            w.writerow([])
            w.writerow(["en_iyi_konum"] + list(self.opt.gbest_x))
        messagebox.showinfo("Kaydedildi", path)
