"""Drive every tab automatically and save screenshots to assets/ (used for the README).

Run with a display, e.g. on Linux CI:  xvfb-run -s "-screen 0 1440x900x24" python tools/screenshots.py
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metalab.gui.app import App  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def shot(app, name):
    app.update()
    app.update_idletasks()
    path = os.path.join(OUT, name)
    wid = app.winfo_id()
    try:
        subprocess.run(["import", "-window", hex(int(app.wm_frame(), 16)) if app.wm_frame() else str(wid),
                        path], check=True)
    except Exception:
        from PIL import ImageGrab
        x, y = app.winfo_rootx(), app.winfo_rooty()
        ImageGrab.grab((x, y, x + app.winfo_width(), y + app.winfo_height())).save(path)
    print("saved", path)


def main():
    os.makedirs(OUT, exist_ok=True)
    app = App()
    app.geometry(os.environ.get("GEOM", "1440x880") + "+0+0")
    nb = app.notebook
    t = app.tabs

    def run_until(tab_opt_getter, cb):
        def check():
            o = tab_opt_getter()
            if o is not None and not o.done:
                app.after(200, check)
            else:
                cb()
        app.after(200, check)

    # 1) visual tab: Rastrigin animation, capture mid-run
    def s1():
        nb.select(0)
        t["visual"].speed.set(100)
        t["visual"].common.set("iters", 60)
        t["visual"].common.set("seed", 3)
        t["visual"].start()
        app.after(1800, lambda: (shot(app, "screen_visual.png"), s2()))

    # 2) teach tab
    def s2():
        t["visual"].running = False
        nb.select(1)
        t["teach"].reset()
        for _ in range(3):
            t["teach"].next_step()
        app.after(300, lambda: (shot(app, "screen_teach.png"), s3()))

    # 3) race tab
    def s3():
        nb.select(2)
        t["race"].common.set("runs", 10)
        t["race"].start()

        def wait():
            if t["race"].worker.is_alive() or str(t["race"].btn.cget("state")) == "disabled":
                app.after(200, wait)
            else:
                app.after(300, lambda: (shot(app, "screen_race.png"), s4()))
        app.after(500, wait)

    # 4) neural tab
    def s4():
        nb.select(3)
        nt = t["neural"]
        nt.speed.set(100)
        nt.start()
        run_until(lambda: nt.pso, lambda: (nt.predict(),
                                           app.after(300, lambda: (shot(app, "screen_neural.png"),
                                                                   app.destroy()))))

    app.after(800, s1)
    app.mainloop()


if __name__ == "__main__":
    main()
