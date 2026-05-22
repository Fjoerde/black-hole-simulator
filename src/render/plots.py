import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from PIL import Image
import io

from Classes.math import *
from Classes.int_and_settings import *
from plots import *


def plot_deviation(geodesics:list[Function], settings:RenderSettings) -> Image.Image:
    """Plots the angular deviation of the light rays across the image."""

    n_px = settings.w * settings.h
    if len(geodesics) != n_px: raise ValueError("Number of geodesics not equal to the number of pixels in the image.")
    if len(geodesics) < 4: return Image.fromarray(np.zeros((1,1,3), dtype=np.uint8))
    deviations = np.empty(n_px, dtype=np.float64)
    for i in range(n_px): deviations[i] = geodesics[i].vals[0,8]
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

def avg_specint(spec_ints:list[Function], settings:RenderSettings) -> Image.Image:
    """Plots the average spectral radiance received by the camera."""

    spectrum = settings.col_converter.grid.pts.flatten()
    flux = np.zeros(len(spectrum), dtype=np.float64)
    for i in range(settings.w * settings.h):
        y, x = i//settings.w, i%settings.w
        cos_theta = settings.ray_dir_px(x, y).dot(settings.cam_dir)
        px_area = (2 * settings.aspect / settings.w) * (2 / settings.h)
        flux += spec_ints[i].vals.flatten() * (px_area * cos_theta / settings.f)**2
    sol_ang = 4 * np.arcsin(settings.aspect / np.sqrt((settings.aspect**2 + settings.f**2)*(1 + settings.f**2)))
    avg_specvals = flux / (4 * settings.aspect * sol_ang)
    avg_specint = Function(settings.col_converter.grid, np.ascontiguousarray(avg_specvals.reshape(len(avg_specvals), 1)))

    xx = np.linspace(np.min(spectrum), np.max(spectrum), 1000)
    yy = avg_specint.interp(xx.reshape(1000, 1))[:,0]
    fig, ax = plt.subplots()
    ax.plot(xx, yy, label="Average Spectral Radiance")
    ax.set_xlabel(r"$\lambda~/~\mathrm{nm}$"); ax.set_ylabel(r"$\langle I_{\lambda}\rangle~/~\mathrm{W~m^{-2}~nm^{-1}}$")
    fig.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    img.show()
    return img
