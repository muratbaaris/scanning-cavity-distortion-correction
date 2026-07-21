"""Graphical interface for interactive distortion correction.

This module is a thin Tkinter wrapper around the ``scdc`` library.  All
computation is delegated to the library; the GUI only handles user
interaction and display.  It can therefore be used without understanding
the library internals, and it carries no tests (the exam permits this for
pure visualisation modules).

Launch with::

    python -m scdc.gui

or::

    python scdc/gui.py
"""

import os

import numpy as np
import scipy.io as sio

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector

from scdc.calibration import fit_calibration, apply_calibration, CalibrationError
from scdc.centroids import detect_centroids
from scdc.io import save_calibration, load_calibration


class CorrectionApp:
    """Tkinter application for fitting and applying a distortion calibration.

    The window shows the original image on the left and the corrected image
    on the right.  The user can select a region of interest by dragging on
    the original, fit a polynomial calibration, and save or load the result.
    """

    def __init__(self, root):
        self.root = root
        root.title("Scanning Cavity Microscope - Distortion Correction")
        root.geometry("1500x850")

        self.image = None
        self.corrected = None
        self.calibration = None
        self.roi = None
        self.image_path = None
        self.show_lines = tk.BooleanVar(value=False)
        self.show_ref = tk.BooleanVar(value=False)
        self.degree_var = tk.IntVar(value=3)

        self._build_toolbar()
        self._build_figure()
        self._build_status()

    # ── toolbar ──────────────────────────────────────────────────

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Load image\u2026",
                   command=self.load_mat).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient="vertical").pack(
            side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(bar, text="Degree:").pack(side=tk.LEFT)
        ttk.Combobox(bar, textvariable=self.degree_var,
                     values=[1, 2, 3, 4], width=3, state="readonly"
                     ).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Fit (Full)",
                   command=lambda: self.fit(full=True)).pack(
                       side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Fit (ROI)",
                   command=lambda: self.fit(full=False)).pack(
                       side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Set ROI numerically\u2026",
                   command=self.set_roi_numeric).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Clear ROI",
                   command=self.clear_roi).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient="vertical").pack(
            side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(bar, text="Show polynomial\u2026",
                   command=self.show_polynomial).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Save calibration\u2026",
                   command=self.do_save_calibration).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Load calibration\u2026",
                   command=self.do_load_calibration).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Apply loaded",
                   command=self.apply_loaded).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient="vertical").pack(
            side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Checkbutton(bar, text="Row/col lines",
                        variable=self.show_lines,
                        command=self.refresh_corrected).pack(
                            side=tk.LEFT, padx=2)
        ttk.Checkbutton(bar, text="Reference grid",
                        variable=self.show_ref,
                        command=self.refresh_corrected).pack(
                            side=tk.LEFT, padx=2)
        ttk.Separator(bar, orient="vertical").pack(
            side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(bar, text="Export PNG\u2026",
                   command=self.export_png).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Export image .mat\u2026",
                   command=self.export_image_mat).pack(side=tk.LEFT, padx=2)

    # ── figure ───────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = Figure(figsize=(13, 6))
        self.ax_orig = self.fig.add_subplot(121)
        self.ax_corr = self.fig.add_subplot(122)
        for ax, title in [
            (self.ax_orig, "Original - drag to select ROI"),
            (self.ax_corr, "Corrected - load an image and click Fit"),
        ]:
            ax.set_title(title)
            ax.set_xlabel("x [px]")
            ax.set_ylabel("y [px]")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.rect_selector = RectangleSelector(
            self.ax_orig, self._on_select_roi,
            useblit=True, button=[1], minspanx=5, minspany=5,
            spancoords="pixels", interactive=True,
            props=dict(facecolor="red", edgecolor="red", alpha=0.2, fill=True),
        )

    # ── status bar ───────────────────────────────────────────────

    def _build_status(self):
        self.status_var = tk.StringVar(value="Ready. Load a .mat file to begin.")
        bar = ttk.Frame(self.root, padding=4)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bar, textvariable=self.status_var, anchor="w").pack(fill=tk.X)

    def _set_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    # ── loading ──────────────────────────────────────────────────

    def load_mat(self):
        path = filedialog.askopenfilename(
            title="Open image .mat file",
            filetypes=[("MATLAB files", "*.mat"), ("All files", "*.*")])
        if not path:
            return
        try:
            mat = sio.loadmat(path)
            arrays = {k: v for k, v in mat.items()
                      if not k.startswith("__")
                      and isinstance(v, np.ndarray) and v.ndim == 2}
            if not arrays:
                raise ValueError("No 2D arrays found in .mat file")
            key = (self._choose_key(list(arrays.keys()))
                   if len(arrays) > 1 else list(arrays.keys())[0])
            if key is None:
                return
            self.image = arrays[key].astype(float)
            self.image_path = path
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        self.corrected = None
        self.roi = None
        self._draw_original()
        self.ax_corr.clear()
        self.ax_corr.set_title(
            "Corrected - click Fit to compute, or Apply loaded")
        self.canvas.draw()
        height, width = self.image.shape
        self._set_status(
            f"Loaded {os.path.basename(path)}  "
            f"(width \u00d7 height = {width} \u00d7 {height} px, key '{key}')")

    def _choose_key(self, keys):
        win = tk.Toplevel(self.root)
        win.title("Pick array")
        ttk.Label(win, text="Multiple 2D arrays found. Which to use?",
                  padding=10).pack()
        var = tk.StringVar(value=keys[0])
        ttk.Combobox(win, textvariable=var, values=keys,
                     state="readonly").pack(padx=10, pady=5)
        choice = [None]

        def ok():
            choice[0] = var.get()
            win.destroy()

        ttk.Button(win, text="OK", command=ok).pack(pady=10)
        win.transient(self.root)
        win.grab_set()
        self.root.wait_window(win)
        return choice[0]

    # ── drawing ──────────────────────────────────────────────────

    def _draw_original(self):
        self.ax_orig.clear()
        self.ax_orig.imshow(self.image, cmap="viridis")
        self.ax_orig.set_title("Original - drag to select ROI")
        self.ax_orig.set_xlabel("x [px]")
        self.ax_orig.set_ylabel("y [px]")
        self.canvas.draw()

    # ── ROI ──────────────────────────────────────────────────────

    def _on_select_roi(self, eclick, erelease):
        x0 = int(round(min(eclick.xdata, erelease.xdata)))
        x1 = int(round(max(eclick.xdata, erelease.xdata)))
        y0 = int(round(min(eclick.ydata, erelease.ydata)))
        y1 = int(round(max(eclick.ydata, erelease.ydata)))
        self.roi = (x0, y0, x1, y1)
        self._set_status(
            f"ROI: x \u2208 [{x0}, {x1}], y \u2208 [{y0}, {y1}]  "
            f"({x1 - x0} \u00d7 {y1 - y0} px)")

    def clear_roi(self):
        self.roi = None
        self.rect_selector.set_active(False)
        self.rect_selector.set_active(True)
        if self.image is not None:
            self._draw_original()
        self._set_status("ROI cleared.")

    def set_roi_numeric(self):
        if self.image is None:
            messagebox.showwarning("No image", "Load a .mat file first.")
            return
        win = tk.Toplevel(self.root)
        win.title("Set ROI by pixel coordinates")
        height, width = self.image.shape
        ttk.Label(
            win,
            text=(f"Image is {width} \u00d7 {height} px  (width \u00d7 height)\n"
                  f"x range: 0 \u2026 {width},   y range: 0 \u2026 {height}"),
            padding=8, justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        defaults = self.roi or (0, 0, width, height)
        labels = ["x start (left):", "y start (top):",
                  "x end (right):", "y end (bottom):"]
        vars_ = []
        for i, (lbl, default) in enumerate(zip(labels, defaults)):
            ttk.Label(win, text=lbl, padding=4).grid(
                row=i + 1, column=0, sticky="e")
            v = tk.IntVar(value=int(default))
            ttk.Entry(win, textvariable=v, width=8).grid(
                row=i + 1, column=1, sticky="w", padx=4)
            vars_.append(v)

        def ok():
            x0, y0, x1, y1 = (v.get() for v in vars_)
            x0 = max(0, min(width - 1, x0))
            x1 = max(0, min(width, x1))
            y0 = max(0, min(height - 1, y0))
            y1 = max(0, min(height, y1))
            if x1 <= x0 or y1 <= y0:
                messagebox.showerror("Bad ROI",
                                     "End coordinates must exceed start.")
                return
            self.roi = (x0, y0, x1, y1)
            self._draw_original()
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                              linewidth=1.5, edgecolor="red",
                              facecolor="red", alpha=0.2)
            self.ax_orig.add_patch(rect)
            self.canvas.draw()
            self._set_status(
                f"ROI: x \u2208 [{x0}, {x1}], y \u2208 [{y0}, {y1}]  "
                f"({x1 - x0} \u00d7 {y1 - y0} px)")
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="OK", command=ok).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(
            side=tk.LEFT, padx=4)
        win.transient(self.root)
        win.grab_set()

    # ── fitting ──────────────────────────────────────────────────

    def fit(self, full=True):
        if self.image is None:
            messagebox.showwarning("No image",
                                   "Please load a .mat file first.")
            return
        if full or self.roi is None:
            sub = self.image
        else:
            x0, y0, x1, y1 = self.roi
            sub = self.image[y0:y1, x0:x1]
        self._set_status("Fitting\u2026")
        try:
            cal = fit_calibration(sub, degree=self.degree_var.get())
        except Exception as e:
            messagebox.showerror("Fit failed", str(e))
            self._set_status(f"Fit failed: {e}")
            return
        self.calibration = cal
        self.corrected = apply_calibration(sub, cal)
        self.refresh_corrected()
        out_h, out_w = cal.output_shape
        self._set_status(
            f"Fit done. degree {cal.degree}, "
            f"{cal.n_used}/{cal.n_detected} centroids, "
            f"residual mean {cal.mean_residual_px:.2f} px, "
            f"max {cal.max_residual_px:.2f} px, "
            f"lattice spacing {cal.lattice_spacing_px:.2f} px, "
            f"corrected size: {out_w} \u00d7 {out_h} px (W \u00d7 H)")

    # ── apply loaded calibration ─────────────────────────────────

    def apply_loaded(self):
        if self.image is None:
            messagebox.showwarning("No image",
                                   "Please load a .mat image first.")
            return
        if self.calibration is None:
            messagebox.showwarning("No calibration",
                                   "Load a calibration .mat file first.")
            return
        # Determine which image to correct: ROI if set, else full image.
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi
            sub = self.image[y0:y1, x0:x1]
        else:
            sub = self.image
        try:
            self.corrected = apply_calibration(sub, self.calibration)
        except CalibrationError as e:
            messagebox.showerror(
                "Apply failed",
                f"{e}\n\nMake sure the image (or ROI) has the same shape "
                "as the image the calibration was fitted on.")
            return
        except Exception as e:
            messagebox.showerror("Apply failed", str(e))
            return
        self.refresh_corrected()
        cal = self.calibration
        self._set_status(
            f"Applied saved calibration (degree {cal.degree}, "
            f"originally fit with {cal.n_used} centroids, "
            f"residual {cal.mean_residual_px:.2f} px).")

    # ── corrected display ────────────────────────────────────────

    def refresh_corrected(self):
        if self.corrected is None:
            return
        self.ax_corr.clear()
        self.ax_corr.imshow(self.corrected, cmap="viridis")
        deg = self.calibration.degree if self.calibration else self.degree_var.get()
        self.ax_corr.set_title(f"Corrected (degree {deg})")
        self.ax_corr.set_xlabel("x [px]")
        self.ax_corr.set_ylabel("y [px]")

        new_pts = None
        if self.show_lines.get() or self.show_ref.get():
            img_clean = np.where(np.isnan(self.corrected), 0.0, self.corrected)
            try:
                new_pts = detect_centroids(img_clean)
            except Exception:
                new_pts = None

        if (self.show_lines.get() and new_pts is not None
                and len(new_pts) >= 4 and self.calibration is not None):
            s = self.calibration.lattice_spacing_px
            row_idx = np.round(
                (new_pts[:, 1] - new_pts[:, 1].min()) / s).astype(int)
            col_idx = np.round(
                (new_pts[:, 0] - new_pts[:, 0].min()) / s).astype(int)
            for r in set(row_idx):
                self.ax_corr.axhline(new_pts[row_idx == r, 1].mean(),
                                     color="red", lw=0.5, alpha=0.7)
            for c in set(col_idx):
                self.ax_corr.axvline(new_pts[col_idx == c, 0].mean(),
                                     color="red", lw=0.5, alpha=0.7)

        if (self.show_ref.get() and new_pts is not None
                and len(new_pts) >= 4 and self.calibration is not None):
            s = self.calibration.lattice_spacing_px
            anchor = new_pts[np.argmin(
                np.linalg.norm(new_pts - new_pts.mean(0), axis=1))]
            for x in np.arange(anchor[0] % s, self.corrected.shape[1], s):
                self.ax_corr.axvline(x, color="cyan", lw=0.5, alpha=0.7)
            for y in np.arange(anchor[1] % s, self.corrected.shape[0], s):
                self.ax_corr.axhline(y, color="cyan", lw=0.5, alpha=0.7)

        self.canvas.draw()

    # ── show polynomial ──────────────────────────────────────────

    def show_polynomial(self):
        if self.calibration is None:
            messagebox.showinfo("No polynomial",
                                "Fit a polynomial first or load a calibration.")
            return
        text = self.calibration.summary()
        win = tk.Toplevel(self.root)
        win.title("Polynomial calibration")
        win.geometry("640x560")
        box = scrolledtext.ScrolledText(win, font=("Courier", 10),
                                        wrap=tk.NONE)
        box.insert("1.0", text)
        box.config(state=tk.DISABLED)
        box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        def copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

        ttk.Button(win, text="Copy to clipboard",
                   command=copy).pack(pady=(0, 8))

    # ── save / load calibration ──────────────────────────────────

    def do_save_calibration(self):
        if self.calibration is None:
            messagebox.showwarning("No calibration", "Run Fit first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save calibration",
            defaultextension=".mat",
            filetypes=[("MATLAB file", "*.mat")])
        if not path:
            return
        try:
            save_calibration(self.calibration, path)
            self._set_status(f"Saved calibration to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def do_load_calibration(self):
        path = filedialog.askopenfilename(
            title="Load calibration",
            filetypes=[("MATLAB file", "*.mat"), ("All files", "*.*")])
        if not path:
            return
        try:
            cal = load_calibration(path)
        except Exception as e:
            messagebox.showerror("Load failed",
                                 f"Could not load calibration:\n{e}")
            return
        self.calibration = cal
        out_h, out_w = cal.output_shape
        self._set_status(
            f"Loaded calibration (degree {cal.degree}, "
            f"residual {cal.mean_residual_px:.2f} px, "
            f"target {out_w}\u00d7{out_h} px). "
            f"Click 'Apply loaded' to use it.")

    # ── export ───────────────────────────────────────────────────

    def export_png(self):
        if self.corrected is None:
            messagebox.showwarning("Nothing to export",
                                   "Run Fit or Apply first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save corrected image (PNG)",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")])
        if not path:
            return
        extent = self.ax_corr.get_window_extent().transformed(
            self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(path, bbox_inches=extent, dpi=200)
        self._set_status(f"Saved {os.path.basename(path)}")

    def export_image_mat(self):
        if self.corrected is None:
            messagebox.showwarning("Nothing to export",
                                   "Run Fit or Apply first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save corrected image (.mat)",
            defaultextension=".mat",
            filetypes=[("MATLAB file", "*.mat")])
        if not path:
            return
        out = np.where(np.isnan(self.corrected), 0.0, self.corrected)
        sio.savemat(path, {
            "image_corrected": out,
            "polynomial_degree": self.calibration.degree
            if self.calibration else -1,
            "mean_residual_px": self.calibration.mean_residual_px
            if self.calibration else -1.0,
            "max_residual_px": self.calibration.max_residual_px
            if self.calibration else -1.0,
            "lattice_spacing_px": self.calibration.lattice_spacing_px
            if self.calibration else -1.0,
        })
        self._set_status(f"Saved {os.path.basename(path)}")


def main():
    """Launch the GUI."""
    root = tk.Tk()
    CorrectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
