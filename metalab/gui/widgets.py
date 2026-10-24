"""Reusable widgets: a validated parameter form and a matplotlib panel."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class ParamForm(ttk.LabelFrame):
    """Labelled entries built from a spec list.

    spec item: (key, label, default, type, (min, max) or None)
    ``type`` is int, float or "seed" (empty string -> None).
    """

    def __init__(self, parent, title, spec, **kw):
        super().__init__(parent, text=title, padding=(10, 6), **kw)
        self.spec = spec
        self.vars = {}
        for row, (key, label, default, _typ, _rng) in enumerate(spec):
            ttk.Label(self, text=label, style="Panel.TLabel").grid(
                row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="" if default is None else str(default))
            ttk.Entry(self, textvariable=var, width=9, justify="right").grid(
                row=row, column=1, sticky="e", pady=2, padx=(6, 0))
            self.vars[key] = var
        self.columnconfigure(0, weight=1)

    def get(self):
        """Return a dict of parsed values or raise ValueError (Turkish message)."""
        out = {}
        for key, label, _default, typ, rng in self.spec:
            raw = self.vars[key].get().strip().replace(",", ".")
            if typ == "seed":
                if raw == "":
                    out[key] = None
                    continue
                typ = int
            try:
                val = typ(float(raw)) if typ is int else typ(raw)
            except ValueError:
                raise ValueError(f"'{label}' için geçerli bir sayı girin.") from None
            if rng is not None and not (rng[0] <= val <= rng[1]):
                raise ValueError(f"'{label}' {rng[0]} ile {rng[1]} arasında olmalı.")
            out[key] = val
        return out

    def set(self, key, value):
        self.vars[key].set(str(value))


class PlotPanel(ttk.Frame):
    """A matplotlib Figure embedded in Tk, with the standard zoom/pan toolbar."""

    def __init__(self, parent, figsize=(8, 6), toolbar=True):
        super().__init__(parent, style="Panel.TFrame")
        self.fig = Figure(figsize=figsize, dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        if toolbar:
            tb = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
            tb.update()
            tb.pack(side="bottom", fill="x")

    def draw(self):
        self.canvas.draw_idle()

    def save_png(self, default_name="grafik.png"):
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=default_name,
            filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Kaydedildi", f"Grafik kaydedildi:\n{path}")


def error(msg):
    messagebox.showerror("Hatalı giriş", msg)


def scroll_column(parent, side="left", padx=(10, 6)):
    """A vertically scrollable column (so the panel also fits small laptop screens).

    Packs itself into ``parent`` and returns the inner frame to put widgets in.
    """
    outer = ttk.Frame(parent)
    outer.pack(side=side, fill="y", padx=padx, pady=10)
    canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0, bg="#f4f5f7")
    bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=bar.set)
    canvas.pack(side="left", fill="y")

    def on_configure(_event=None):
        canvas.configure(width=inner.winfo_reqwidth(), scrollregion=canvas.bbox("all"))
        if inner.winfo_reqheight() > canvas.winfo_height() > 1:
            bar.pack(side="left", fill="y")
        else:
            bar.pack_forget()

    inner.bind("<Configure>", on_configure)
    canvas.bind("<Configure>", on_configure)

    def wheel(event):
        if inner.winfo_reqheight() <= canvas.winfo_height():
            return
        delta = -1 if (event.num == 4 or event.delta > 0) else 1
        canvas.yview_scroll(delta, "units")

    def bind_wheel(_e):
        canvas.bind_all("<MouseWheel>", wheel)
        canvas.bind_all("<Button-4>", wheel)
        canvas.bind_all("<Button-5>", wheel)

    def unbind_wheel(_e):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.unbind_all(seq)

    outer.bind("<Enter>", bind_wheel)
    outer.bind("<Leave>", unbind_wheel)
    return inner


def stat_row(parent, row, label):
    """A 'label : value' line in a results panel; returns the value StringVar."""
    ttk.Label(parent, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=1)
    var = tk.StringVar(value="–")
    ttk.Label(parent, textvariable=var, style="Value.TLabel").grid(
        row=row, column=1, sticky="e", pady=1, padx=(8, 0))
    return var


PSO_SPEC = [
    ("w_max", "w başlangıç (ataletsel ağırlık)", 0.9, float, (0, 2)),
    ("w_min", "w bitiş", 0.4, float, (0, 2)),
    ("c1", "c1 (bilişsel / bireysel)", 1.494, float, (0, 4)),
    ("c2", "c2 (sosyal / sürü)", 1.494, float, (0, 4)),
    ("vmax", "Vmax (aralığın oranı)", 0.2, float, (0.01, 2)),
]

GA_SPEC = [
    ("pc", "Çaprazlama oranı", 0.9, float, (0, 1)),
    ("pm", "Mutasyon oranı (gen başı)", 0.1, float, (0, 1)),
    ("sigma", "Mutasyon şiddeti", 0.1, float, (0.001, 1)),
    ("elitism", "Elit birey sayısı", 2, int, (0, 50)),
]

SELECTION_OPTIONS = {"Turnuva (k=3)": "tournament", "Rulet tekerleği": "roulette"}
