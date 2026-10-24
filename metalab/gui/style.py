"""Colours, fonts and ttk theme shared by every tab."""

import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")

BG = "#f4f5f7"
PANEL = "#ffffff"
BORDER = "#dde1e7"
TEXT = "#1f2933"
MUTED = "#616e7c"
ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
HEADER = "#111827"

PSO_COLOR = "#2563eb"
GA_COLOR = "#ea580c"
BP_COLOR = "#7c3aed"
GOOD = "#15803d"
BAD = "#b91c1c"

FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_MONO = ("Consolas", 10)


def apply_theme(root):
    root.configure(bg=BG)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=BG, foreground=TEXT, font=FONT)
    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Panel.TLabel", background=PANEL)
    style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
    style.configure("Value.TLabel", background=PANEL, font=FONT_BOLD)
    style.configure("Big.TLabel", background=PANEL, font=("Segoe UI", 14, "bold"))
    style.configure("TLabelframe", background=PANEL, bordercolor=BORDER, relief="solid")
    style.configure("TLabelframe.Label", background=PANEL, foreground=ACCENT, font=FONT_BOLD)
    style.configure("TCheckbutton", background=PANEL)
    style.configure("TRadiobutton", background=PANEL)
    style.configure("TEntry", fieldbackground="#ffffff", bordercolor=BORDER, padding=3)
    style.configure("TCombobox", fieldbackground="#ffffff", padding=3)
    style.configure("TButton", padding=(10, 6), background="#e5e7eb", bordercolor=BORDER)
    style.map("TButton", background=[("active", "#d1d5db")])
    style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                    font=FONT_BOLD, bordercolor=ACCENT_DARK)
    style.map("Accent.TButton", background=[("active", ACCENT_DARK), ("disabled", "#93c5fd")])
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(16, 8), font=FONT_BOLD, background="#e5e7eb")
    style.map("TNotebook.Tab", background=[("selected", PANEL)],
              foreground=[("selected", ACCENT)])
    style.configure("Treeview", rowheight=22, fieldbackground=PANEL, background=PANEL)
    style.configure("Treeview.Heading", font=FONT_BOLD)
    style.configure("Horizontal.TProgressbar", background=ACCENT)

    matplotlib.rcParams.update({
        "figure.facecolor": PANEL, "axes.facecolor": PANEL,
        "axes.edgecolor": "#9aa5b1", "axes.titleweight": "bold",
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.grid": True,
        "grid.color": "#e4e7eb", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return style


def make_text(parent, height=10):
    """Read-only monospace text box with a scrollbar."""
    frame = ttk.Frame(parent, style="Panel.TFrame")
    text = tk.Text(frame, height=height, font=FONT_MONO, wrap="none", bg="#fbfbfc",
                   fg=TEXT, relief="flat", borderwidth=0, padx=8, pady=6,
                   highlightthickness=1, highlightbackground=BORDER)
    sy = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    sx = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
    text.grid(row=0, column=0, sticky="nsew")
    sy.grid(row=0, column=1, sticky="ns")
    sx.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    text.tag_configure("h", font=("Consolas", 10, "bold"), foreground=ACCENT)
    text.tag_configure("good", foreground=GOOD, font=("Consolas", 10, "bold"))
    text.tag_configure("muted", foreground=MUTED)
    return frame, text
