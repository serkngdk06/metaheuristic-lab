"""Tab 2 - teaching mode: the lecture's PSO example, solved step by step.

Reproduces the course's psoVisual.py problem (find x with a*x^2 + b*x + c = 0)
and prints every term of the velocity formula with real numbers, the same
way the hand calculation in the lecture notes is written.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.ticker import MaxNLocator

from ..algorithms import PSO
from . import style as S
from .style import make_text as make_text_box
from .widgets import ParamForm, PlotPanel, error, scroll_column


class TeachTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.opt = None
        self._job = None
        self._build()
        self.reset()

    def _build(self):
        left = scroll_column(self)

        prob = ttk.LabelFrame(left, text="Problem:  a·x² + b·x + c = 0", padding=(10, 6))
        prob.pack(fill="x")
        self.coef = {}
        for i, (k, v) in enumerate([("a", 1), ("b", 2), ("c", -3)]):
            ttk.Label(prob, text=f"{k} =", style="Panel.TLabel").grid(row=0, column=2 * i, padx=(0, 2))
            var = tk.StringVar(value=str(v))
            ttk.Entry(prob, textvariable=var, width=5, justify="right").grid(
                row=0, column=2 * i + 1, padx=(0, 8))
            self.coef[k] = var
        ttk.Label(prob, text="Başlangıç parçacıkları (virgül veya boşluk)",
                  style="Panel.TLabel").grid(row=1, column=0, columnspan=6, sticky="w", pady=(6, 0))
        self.init_var = tk.StringVar(value="3 7 5")
        ttk.Entry(prob, textvariable=self.init_var).grid(row=2, column=0, columnspan=6, sticky="ew")

        self.form = ParamForm(left, "PSO parametreleri", [
            ("w", "w (ataletsel ağırlık)", 0.9, float, (0, 2)),
            ("c1", "c1", 1.494, float, (0, 4)),
            ("c2", "c2", 1.494, float, (0, 4)),
            ("iters", "İterasyon sayısı", 30, int, (1, 500)),
        ])
        self.form.pack(fill="x", pady=(8, 0))

        rbox = ttk.LabelFrame(left, text="r1, r2 rastgele sayıları", padding=(10, 6))
        rbox.pack(fill="x", pady=(8, 0))
        self.r_mode = tk.StringVar(value="fixed")
        ttk.Radiobutton(rbox, text="Sabit 0.5 (ders örneğindeki gibi)", value="fixed",
                        variable=self.r_mode).pack(anchor="w")
        ttk.Radiobutton(rbox, text="Her adımda rastgele [0, 1)  (standart PSO)",
                        value="random", variable=self.r_mode).pack(anchor="w")

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(10, 0))
        ttk.Button(btns, text="↺  Başlat / Sıfırla", style="Accent.TButton",
                   command=self.reset).pack(fill="x")
        ttk.Button(btns, text="Sonraki adım  ▶", command=self.next_step).pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="Sona kadar oynat  ⏭", command=self.play).pack(fill="x", pady=(4, 0))

        note = ttk.LabelFrame(left, text="Not", padding=(10, 6))
        note.pack(fill="x", pady=(8, 0))
        ttk.Label(note, style="Muted.TLabel", wraplength=245, justify="left", text=(
            "Uygunluk |f(x)| olarak alınır: amaç f(x)'i 0'a yaklaştırmaktır. "
            "Bu yüzden gbest seçilirken en küçük f(x) değil, 0'a en yakın "
            "(mutlak değeri en küçük) f(x) değeri kullanılır.\n\n"
            "Formül:  v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)\n"
            "             x ← x + v")).pack(anchor="w")

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
        self.plot = PlotPanel(right, figsize=(8, 3.3), toolbar=False)
        self.plot.pack(fill="x")
        gs = self.plot.fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.25)
        self.ax = self.plot.fig.add_subplot(gs[0])
        self.ax_c = self.plot.fig.add_subplot(gs[1])
        self.plot.fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.16)

        box, self.log = make_text_box(right, height=18)
        box.pack(fill="both", expand=True, pady=(8, 0))

    # ------------------------------------------------------------------ #
    def _f(self, x):
        return self.a * x ** 2 + self.b * x + self.c

    def reset(self):
        if self._job:
            self.after_cancel(self._job)
            self._job = None
        try:
            self.a, self.b, self.c = (float(self.coef[k].get().replace(",", ".")) for k in "abc")
            raw = self.init_var.get().replace(",", " ").split()
            init = [float(v) for v in raw]
            p = self.form.get()
        except ValueError as exc:
            msg = str(exc) if "'" in str(exc) else "Katsayılar ve parçacıklar sayı olmalı."
            error(msg)
            return
        if not 1 <= len(init) <= 20:
            error("1 ile 20 arasında başlangıç parçacığı girin.")
            return

        fitness = lambda P: np.abs(self._f(P[:, 0]))  # noqa: E731
        span = max(50.0, 4 * max(abs(v) for v in init))
        self.opt = PSO(fitness, 1, (-span, span), max_iter=p["iters"],
                       w_max=p["w"], w_min=p["w"], c1=p["c1"], c2=p["c2"],
                       v_max_ratio=None, clip_to_bounds=False,
                       fixed_r=0.5 if self.r_mode.get() == "fixed" else None,
                       init_positions=[[v] for v in init])
        self.xs_hist = [self.opt.x[:, 0].copy()]

        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        o = self.opt
        self._write(f"f(x) = {self.a:g}·x² + {self.b:+g}·x + {self.c:+g}     "
                    f"w={p['w']:g}  c1={p['c1']:g}  c2={p['c2']:g}\n", "h")
        self._write("İterasyon 0 — başlangıç değerlendirmesi\n", "h")
        for i in range(o.n):
            x = o.x[i, 0]
            self._write(f"  P{i + 1}: x = {x:10.4f}   f(x) = {self._f(x):12.4f}   "
                        f"pbest = {x:.4f}\n")
        self._write(f"  → gbest = {o.gbest_x[0]:.4f}   (f = {self._f(o.gbest_x[0]):.4f})\n\n",
                    "good")
        self.log.config(state="disabled")
        self._draw()

    def _write(self, text, tag=None):
        self.log.insert("end", text, tag)

    def next_step(self):
        if self.opt is None:
            return
        if self.opt.done:
            self._job = None
            return
        d = self.opt.step()
        o = self.opt
        self.xs_hist.append(o.x[:, 0].copy())
        self.log.config(state="normal")
        self._write(f"İterasyon {d['iteration']}   (gbest = {d['gbest_used'][0]:.4f})\n", "h")
        for i in range(o.n):
            x, v = d["x_old"][i, 0], d["v_old"][i, 0]
            pb, gb = d["pbest_used"][i, 0], d["gbest_used"][0]
            r1, r2 = d["r1"][i, 0], d["r2"][i, 0]
            vn, xn = d["v_new"][i, 0], d["x_new"][i, 0]
            self._write(
                f"  P{i + 1}: v = {d['w']:.3f}·{v:.4f} + {o.c1:g}·{r1:.3f}·({pb:.4f} − {x:.4f})"
                f" + {o.c2:g}·{r2:.3f}·({gb:.4f} − {x:.4f}) = {vn:.4f}\n")
            mark = "   ★ pbest güncellendi" if d["improved"][i] else ""
            self._write(f"       x = {x:.4f} + ({vn:.4f}) = {xn:.4f}    "
                        f"f(x) = {self._f(xn):.4f}{mark}\n", "muted")
        self._write(f"  → gbest = {o.gbest_x[0]:.6f}   |f(gbest)| = {o.gbest_f:.6f}\n\n", "good")
        if o.done:
            self._write(f"Bitti: {o.iteration} iterasyon sonunda x* = {o.gbest_x[0]:.6f}, "
                        f"f(x*) = {self._f(o.gbest_x[0]):.6f}\n", "h")
        self.log.config(state="disabled")
        self.log.see("end")
        self._draw()

    def play(self):
        if self.opt is None or self.opt.done:
            return
        self.next_step()
        self._job = self.after(250, self.play)

    def _draw(self):
        o = self.opt
        ax = self.ax
        ax.clear()
        hist = np.array(self.xs_hist)
        pts = np.concatenate([hist.ravel(), [o.gbest_x[0]]])
        lo, hi = pts.min(), pts.max()
        pad = max(1.0, 0.15 * (hi - lo))
        xs = np.linspace(lo - pad, hi + pad, 500)
        ax.plot(xs, self._f(xs), color="#334e68", lw=2, label="f(x)")
        ax.axhline(0, color="#9aa5b1", lw=1)
        disc = self.b ** 2 - 4 * self.a * self.c
        if self.a != 0 and disc >= 0:
            roots = [(-self.b + s * np.sqrt(disc)) / (2 * self.a) for s in (1, -1)]
            ax.scatter(roots, [0, 0], marker="x", s=60, c=S.GOOD, zorder=4, label="gerçek kökler")
        colors = ["#2563eb", "#ea580c", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]
        for i in range(o.n):
            col = colors[i % len(colors)]
            ax.scatter(o.x[i, 0], self._f(o.x[i, 0]), s=80, color=col, edgecolors="white",
                       zorder=5, label=f"P{i + 1}" if i < 6 else None)
            if len(hist) > 1:
                ax.plot(hist[-2:, i], self._f(hist[-2:, i]), color=col, lw=1, alpha=0.5)
        ax.scatter(o.gbest_x[0], self._f(o.gbest_x[0]), s=260, marker="*", c="#facc15",
                   edgecolors="black", zorder=6, label="gbest")
        ax.set_title(f"İterasyon {o.iteration}/{o.max_iter}  •  gbest x = {o.gbest_x[0]:.4f}")
        ax.set_xlabel("x")
        ax.legend(loc="upper center", ncol=4, fontsize=7)

        c = self.ax_c
        c.clear()
        c.plot(hist, marker="o", ms=3)
        c.xaxis.set_major_locator(MaxNLocator(integer=True))
        c.set_title("Parçacıkların konumu")
        c.set_xlabel("iterasyon")
        c.set_ylabel("x")
        self.plot.draw()
