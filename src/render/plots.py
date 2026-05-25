import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from PIL import Image
import io

from Classes.math import *
from Classes.int_and_settings import *

def deviation(geodesics:list[Function], settings:RenderSettings) -> Image.Image:
    """Plots the angular deviation of the light rays across the image."""

    n_px = settings.w * settings.h
    if len(geodesics) != n_px: raise ValueError("Number of geodesics not equal to the number of pixels in the image.")
    deviations = np.empty(n_px, dtype=np.float64)
    for i in range(n_px):
        k_i = geodesics[i].vals[0,5:]; k_f = geodesics[i].vals[-1,5:]
        deviations[i] = np.arccos(np.clip((k_i @ k_f) / (np.linalg.norm(k_i) * np.linalg.norm(k_f)), -1, 1))
    deviations = deviations.reshape(settings.h, settings.w)

    fig, ax = plt.subplots()
    X = np.arange(settings.w); Y = np.flip(settings.h-1 - np.arange(settings.h))
    xx, yy = np.meshgrid(X, Y)
    contourf = ax.contourf(xx, yy, deviations, levels=50, cmap="viridis",
                           norm=PowerNorm(gamma=0.3, vmin=0, vmax=np.max(deviations)))
    fig.colorbar(contourf, ax=ax, label=r"$\theta_{d}~/~\mathrm{rad}$")
    ax.set_title("Angular Deviation")
    ax.set_aspect("equal")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    img.show()
    return img

def tot_specint(spec_ints:list[Function], settings:RenderSettings) -> Image.Image:
    """Plots the average spectral radiance received by the camera."""

    spectrum = settings.col_converter.grid.pts.flatten()
    tot_specvals = np.zeros(len(spectrum), dtype=np.float64)
    for i in range(settings.w * settings.h):
        y, x = i//settings.w, i%settings.w
        ray_dir = settings.ray_dir_px(x, y)
        px_x, px_y = 2 * settings.aspect / settings.w, 2 / settings.h
        theta = np.arccos(ray_dir.dot(settings.cam_dir))
        r = settings.f / np.cos(theta); sol_ang_px = px_x * px_y * np.cos(theta) / r**2
        tot_specvals += spec_ints[i].vals.flatten() * sol_ang_px
    
    tot_specint = Function(settings.col_converter.grid, np.ascontiguousarray(tot_specvals.reshape(len(tot_specvals), 1)))
    xx = np.linspace(np.min(spectrum), np.max(spectrum), 1000)
    yy = tot_specint.interp(xx.reshape(1000, 1))[:,0]
    fig, ax = plt.subplots()
    ax.plot(xx, yy, label="Total Spectral Intensity")
    ax.set_xlabel(r"$\lambda~/~\mathrm{nm}$"); ax.set_ylabel(r"$I_{\lambda}~/~\mathrm{W~m^{-2}~nm^{-1}}$")
    fig.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    img.show()
    return img
