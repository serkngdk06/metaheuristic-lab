"""Tab 4 - hybrid approach: training a neural network's weights with PSO.

Every particle is a complete set of network weights. The swarm searches the
weight space and the fitness of a particle is the network's training MSE.
The decision boundary, the network diagram and the loss curve are redrawn
live while the swarm learns. Backpropagation can be run side by side.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np

from ..algorithms import PSO
from ..neural import DATASETS, MLP, BackpropTrainer, make_student_dataset, normalize
from . import style as S
from .widgets import ParamForm, PlotPanel, error, stat_row, scroll_column

GRID_N = 90


class NeuralTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.data = None
        self.pso = None
        self.bp = None
        self.running = False
        self._job = None
        g = np.linspace(0, 100, GRID_N)
        self.gx, self.gy = np.meshgrid(g, g)
        self.grid_inputs = normalize(np.column_stack([self.gx.ravel(), self.gy.ravel()]))
        self._build()
        self.generate_data()

    # ------------------------------------------------------------ UI --- #
    def _build(self):
        left = scroll_column(self)

        dbox = ttk.LabelFrame(left, text="1) Veri seti: öğrenci notları", padding=(10, 6))
        dbox.pack(fill="x")
        self.kind_var = tk.StringVar(value=list(DATASETS.values())[0])
        ttk.Combobox(dbox, textvariable=self.kind_var, state="readonly", width=34,
                     values=list(DATASETS.values())).pack(fill="x")
        self.data_form = ParamForm(left, "Veri ayarları", [
            ("n", "Öğrenci sayısı", 200, int, (20, 2000)),
            ("noise", "Gürültü (puan)", 4, float, (0, 30)),
            ("seed", "Seed (boş = rastgele)", 7, "seed", None),
        ])
        self.data_form.pack(fill="x", pady=(6, 0))
        ttk.Button(left, text="Veri üret", command=self.generate_data).pack(fill="x", pady=(4, 0))

        self.net_form = ParamForm(left, "2) Ağ ve PSO", [
            ("hidden", "Gizli nöron sayısı", 2, int, (1, 12)),
            ("n", "Parçacık sayısı", 30, int, (2, 300)),
            ("iters", "İterasyon sayısı", 300, int, (1, 5000)),
            ("w_max", "w başlangıç", 0.9, float, (0, 2)),
            ("w_min", "w bitiş", 0.4, float, (0, 2)),
            ("c1", "c1", 1.494, float, (0, 4)),
            ("c2", "c2", 1.494, float, (0, 4)),
            ("bound", "Ağırlık aralığı ±", 10, float, (0.1, 100)),
        ])
        self.net_form.pack(fill="x", pady=(8, 0))

        bpbox = ttk.LabelFrame(left, text="3) Karşılaştırma", padding=(10, 6))
        bpbox.pack(fill="x", pady=(8, 0))
        self.use_bp = tk.BooleanVar(value=True)
        ttk.Checkbutton(bpbox, text="Geri yayılım ile de eğit", variable=self.use_bp).grid(
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(bpbox, text="Öğrenme oranı (η)", style="Panel.TLabel").grid(row=1, column=0, sticky="w")
        self.lr_var = tk.StringVar(value="5")
        ttk.Entry(bpbox, textvariable=self.lr_var, width=9, justify="right").grid(
            row=1, column=1, sticky="e", pady=2)
        bpbox.columnconfigure(0, weight=1)

        self.train_btn = ttk.Button(left, text="🧠  Eğitimi başlat", style="Accent.TButton",
                                    command=self.start)
        self.train_btn.pack(fill="x", pady=(10, 0))
        row = ttk.Frame(left)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="⏸ Durdur / Devam", command=self.toggle).pack(
            side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(row, text="PNG", command=lambda: self.plot.save_png("pso_ysa.png")).pack(
            side="left", padx=(2, 0))
        self.speed = tk.IntVar(value=60)
        sp = ttk.Frame(left)
        sp.pack(fill="x", pady=(6, 0))
        ttk.Label(sp, text="Animasyon hızı").pack(side="left")
        ttk.Scale(sp, from_=1, to=100, variable=self.speed).pack(side="left", fill="x",
                                                                expand=True, padx=4)

        # ---- centre: plots
        self.plot = PlotPanel(self, figsize=(8.4, 7))   # packed last (fills the rest)
        f = self.plot.fig
        gs = f.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.38, wspace=0.28)
        self.ax_pso = f.add_subplot(gs[0, 0])
        self.ax_net = f.add_subplot(gs[0, 1])
        self.ax_loss = f.add_subplot(gs[1, 0])
        self.ax_bp = f.add_subplot(gs[1, 1])
        f.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.08)

        # ---- right: results + try it
        right = scroll_column(self, side="right", padx=(6, 10))
        res = ttk.LabelFrame(right, text="Sonuçlar", padding=(10, 6))
        res.pack(fill="x")
        ttk.Label(res, text="PSO", style="Value.TLabel", foreground=S.PSO_COLOR).grid(
            row=0, column=0, sticky="w")
        self.v_iter = stat_row(res, 1, "İterasyon")
        self.v_loss = stat_row(res, 2, "Eğitim hatası (MSE)")
        self.v_tr = stat_row(res, 3, "Eğitim doğruluğu")
        self.v_te = stat_row(res, 4, "Test doğruluğu")
        ttk.Label(res, text="Geri yayılım", style="Value.TLabel", foreground=S.BP_COLOR).grid(
            row=5, column=0, sticky="w", pady=(6, 0))
        self.v_bp_loss = stat_row(res, 6, "Eğitim hatası (MSE)")
        self.v_bp_te = stat_row(res, 7, "Test doğruluğu")
        self.v_cost = ttk.Label(res, text="", style="Muted.TLabel", wraplength=225, justify="left")
        self.v_cost.grid(row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))
        res.columnconfigure(1, weight=1)

        wbox = ttk.LabelFrame(right, text="PSO'nun bulduğu ağırlıklar (gbest)", padding=(6, 6))
        wbox.pack(fill="both", expand=True, pady=(8, 0))
        self.wtable = ttk.Treeview(wbox, columns=("p", "v"), show="headings", height=9)
        self.wtable.heading("p", text="Parametre")
        self.wtable.heading("v", text="Değer")
        self.wtable.column("p", width=80, anchor="center")
        self.wtable.column("v", width=95, anchor="e")
        sb = ttk.Scrollbar(wbox, orient="vertical", command=self.wtable.yview)
        self.wtable.configure(yscrollcommand=sb.set)
        self.wtable.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

        tbox = ttk.LabelFrame(right, text="4) Kendin dene", padding=(10, 6))
        tbox.pack(fill="x", pady=(8, 0))
        self.try_exam = tk.StringVar(value="45")
        self.try_proj = tk.StringVar(value="70")
        ttk.Label(tbox, text="Sınav notu", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(tbox, textvariable=self.try_exam, width=7, justify="right").grid(row=0, column=1)
        ttk.Label(tbox, text="Proje notu", style="Panel.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Entry(tbox, textvariable=self.try_proj, width=7, justify="right").grid(row=1, column=1)
        ttk.Button(tbox, text="Tahmin et", command=self.predict).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.pred_label = ttk.Label(tbox, text="–", style="Big.TLabel")
        self.pred_label.grid(row=3, column=0, columnspan=2, pady=(6, 0))
        self.pred_detail = ttk.Label(tbox, text="", style="Muted.TLabel", wraplength=220)
        self.pred_detail.grid(row=4, column=0, columnspan=2)
        tbox.columnconfigure(0, weight=1)

        self.plot.pack(side="left", fill="both", expand=True, pady=10)

    # ---------------------------------------------------------- data --- #
    def generate_data(self):
        try:
            d = self.data_form.get()
        except ValueError as exc:
            error(str(exc))
            return
        kind = next(k for k, v in DATASETS.items() if v == self.kind_var.get())
        Xtr, ytr, Xte, yte = make_student_dataset(d["n"], d["noise"], kind, seed=d["seed"])
        self.data = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
                     "Xtr_n": normalize(Xtr), "Xte_n": normalize(Xte)}
        self._stop()
        self.pso = self.bp = None
        self._draw()

    # ------------------------------------------------------ training --- #
    def start(self):
        if self.data is None:
            return
        try:
            p = self.net_form.get()
            lr = float(self.lr_var.get().replace(",", "."))
        except ValueError as exc:
            error(str(exc) if "'" in str(exc) else "Öğrenme oranı sayı olmalı.")
            return
        self._stop()
        d = self.data
        self.mlp = MLP(p["hidden"])
        fitness = self.mlp.fitness_function(d["Xtr_n"], d["ytr"])
        self.pso = PSO(fitness, self.mlp.n_params, (-p["bound"], p["bound"]), p["n"], p["iters"],
                       w_max=p["w_max"], w_min=p["w_min"], c1=p["c1"], c2=p["c2"],
                       v_max_ratio=0.2)
        self.bp = BackpropTrainer(self.mlp, d["Xtr_n"], d["ytr"], lr=lr) if self.use_bp.get() else None
        self.running = True
        self._tick()

    def _tick(self):
        if not self.running or self.pso is None:
            return
        steps = max(1, int(np.interp(self.speed.get(), [1, 100], [1, 12])))
        for _ in range(steps):
            if self.pso.done:
                break
            self.pso.step()
            if self.bp is not None:
                self.bp.step()   # 1 epoch per PSO iteration -> same x-axis
        self._draw()
        if self.pso.done:
            self.running = False
            return
        delay = int(np.interp(self.speed.get(), [1, 100], [250, 15]))
        self._job = self.after(delay, self._tick)

    def toggle(self):
        if self.pso is None or self.pso.done:
            return
        self.running = not self.running
        if self.running:
            self._tick()

    def _stop(self):
        self.running = False
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None

    # ------------------------------------------------------- drawing --- #
    def _boundary(self, ax, params, title, color):
        d = self.data
        ax.clear()
        ax.grid(False)
        if params is not None:
            prob = self.mlp.forward(self.grid_inputs, params).reshape(self.gx.shape)
            ax.imshow(prob, origin="lower", extent=(0, 100, 0, 100), cmap="RdYlGn",
                      vmin=0, vmax=1, alpha=0.55, aspect="auto")
            ax.contour(self.gx, self.gy, prob, levels=[0.5], colors=[color], linewidths=2)
        for X, y, marker, lab in ((d["Xtr"], d["ytr"], "o", "eğitim"),
                                  (d["Xte"], d["yte"], "^", "test")):
            ax.scatter(X[y == 1, 0], X[y == 1, 1], s=16, marker=marker, c="#15803d",
                       edgecolors="white", linewidths=0.4, label=f"geçti ({lab})")
            ax.scatter(X[y == 0, 0], X[y == 0, 1], s=16, marker=marker, c="#b91c1c",
                       edgecolors="white", linewidths=0.4, label=f"kaldı ({lab})")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("sınav notu")
        ax.set_ylabel("proje notu")
        ax.set_title(title, color=color)

    def _draw_network(self, ax, params):
        ax.clear()
        ax.axis("off")
        ax.set_title("Ağ yapısı  (mavi +, kırmızı −)")
        H = self.mlp.h if self.pso is not None else int(self.net_form.vars["hidden"].get() or 2)
        layers = [2, H, 1]
        pos = {}
        for li, n in enumerate(layers):
            ys = np.linspace(0.85, 0.15, n) if n > 1 else [0.5]
            for ni, yv in enumerate(ys):
                pos[(li, ni)] = (0.12 + li * 0.38, yv)
        if params is not None:
            W1, b1, W2, b2 = self.mlp.decode(params)
            m = max(np.abs(W1).max(), np.abs(W2).max(), 1e-9)
            for i in range(2):
                for j in range(H):
                    self._edge(ax, pos[(0, i)], pos[(1, j)], W1[i, j], m)
            for j in range(H):
                self._edge(ax, pos[(1, j)], pos[(2, 0)], W2[j], m)
        names = {(0, 0): "Sınav", (0, 1): "Proje", (2, 0): "Çıkış"}
        for key, (x, y) in pos.items():
            col = {0: "#e0e7ff", 1: "#fef3c7", 2: "#dcfce7"}[key[0]]
            ax.scatter(x, y, s=1500, c=col, edgecolors="#334e68", linewidths=1.5, zorder=3)
            ax.text(x, y, names.get(key, f"h{key[1] + 1}"), ha="center", va="center",
                    fontsize=8, fontweight="bold", zorder=4)
        n_params = (4 * H + 1)
        ax.text(0.5, 0.0, f"1 parçacık = {n_params} ağırlık\n(2×{H} + {H} + {H} + 1)",
                ha="center", fontsize=8, color=S.MUTED)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1)

    @staticmethod
    def _edge(ax, a, b, w, m):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#2563eb" if w >= 0 else "#dc2626",
                lw=0.5 + 5 * abs(w) / m, alpha=0.75, zorder=1)

    def _draw(self):
        d = self.data
        pso_params = self.pso.gbest_x if self.pso is not None else None
        it = f"iter. {self.pso.iteration}/{self.pso.max_iter}" if self.pso else "eğitilmedi"
        self._boundary(self.ax_pso, pso_params, f"PSO ile eğitim — {it}", S.PSO_COLOR)
        self.ax_pso.legend(loc="lower left", fontsize=6.5, ncol=2, framealpha=0.85)
        self._draw_network(self.ax_net, pso_params)

        bp_params = self.bp.p if self.bp is not None else None
        bp_title = (f"Geri yayılım — epoch {self.bp.epoch}" if self.bp is not None
                    else "Geri yayılım (kapalı)")
        self._boundary(self.ax_bp, bp_params, bp_title, S.BP_COLOR)

        a = self.ax_loss
        a.clear()
        if self.pso is not None:
            a.semilogy(self.pso.history, color=S.PSO_COLOR, lw=2, label="PSO (gbest)")
            if self.bp is not None:
                a.semilogy(self.bp.history, color=S.BP_COLOR, lw=2, label="Geri yayılım")
            a.set_xlim(0, self.pso.max_iter)
            a.legend()
        a.set_title("Eğitim hatası (MSE)")
        a.set_xlabel("iterasyon / epoch")
        a.set_ylabel("MSE [log]")
        self.plot.draw()

        if self.pso is not None:
            p = self.pso.gbest_x
            self.v_iter.set(f"{self.pso.iteration} / {self.pso.max_iter}")
            self.v_loss.set(f"{self.pso.gbest_f:.5f}")
            self.v_tr.set(f"%{100 * self.mlp.accuracy(d['Xtr_n'], d['ytr'], p):.1f}")
            self.v_te.set(f"%{100 * self.mlp.accuracy(d['Xte_n'], d['yte'], p):.1f}")
            self.wtable.delete(*self.wtable.get_children())
            for name, val in zip(self.mlp.param_names(), p):
                self.wtable.insert("", "end", values=(name, f"{val:+.4f}"))
            self.v_cost.config(text=(
                f"Maliyet: PSO {self.pso.evaluations} ileri yayılım yaptı; geri yayılım "
                f"{self.bp.epoch if self.bp else 0} ileri + geri geçiş. PSO türev "
                f"kullanmaz ama daha çok hesap yapar."))
        else:
            for v in (self.v_iter, self.v_loss, self.v_tr, self.v_te):
                v.set("–")
        if self.bp is not None:
            self.v_bp_loss.set(f"{self.bp.history[-1]:.5f}")
            self.v_bp_te.set(f"%{100 * self.mlp.accuracy(d['Xte_n'], d['yte'], self.bp.p):.1f}")
        else:
            self.v_bp_loss.set("–")
            self.v_bp_te.set("–")

    # ------------------------------------------------------- predict --- #
    def predict(self):
        if self.pso is None:
            error("Önce ağı eğitin.")
            return
        try:
            exam = float(self.try_exam.get().replace(",", "."))
            proj = float(self.try_proj.get().replace(",", "."))
        except ValueError:
            error("Notlar sayı olmalı.")
            return
        if not (0 <= exam <= 100 and 0 <= proj <= 100):
            error("Notlar 0 ile 100 arasında olmalı.")
            return
        x = normalize([[exam, proj]])
        prob = float(self.mlp.forward(x, self.pso.gbest_x)[0])
        ok = prob >= 0.5
        self.pred_label.config(text="✔ GEÇTİ" if ok else "✘ KALDI",
                               foreground=S.GOOD if ok else S.BAD)
        self.pred_detail.config(text=f"Ağın çıkışı: {prob:.3f}  "
                                     f"(geçme olasılığı %{100 * prob:.1f})")
        self.ax_pso.scatter([exam], [proj], s=220, marker="P", c="#facc15",
                            edgecolors="black", zorder=7)
        self.plot.draw()
