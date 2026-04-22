import math
import random

import numpy
import matplotlib.pyplot as plt

from simulation_runnable_partb import (
    RFresnel, Hop, Spin, Roulette, StepSizeInTissue, CrossUpOrNot
)

# --- Physical parameters (paper Table 1 / §2.C) ---
MU_S = 100.0  # scattering coefficient, cm⁻¹
G = 0.9  # anisotropy factor
MU_A = 0.1  # absorption coefficient, cm⁻¹
N = 1.33  # refractive index of medium (water-like; matched at boundaries)
NA = 0.1  # numerical aperture of objective lens (in air)
LAMBDA = 5.70e-5  # wavelength, cm (570 nm)

# Derived beam parameters (corrected for paper's stated ω₀; see plan)
OMEGA0 = LAMBDA / (math.pi * NA * N)
Z0 = math.pi * OMEGA0**2 * N / LAMBDA

# Transport mean free path
L_T_PRIME = 1.0 / (MU_S * (1.0 - G) + MU_A)

ZF_OVER_LT = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 3.0]

DR = 1e-5  # cm = 0.1 μm (radial bin width)
DZ = 1e-4  # cm = 1 μm (depth bin width)
NR = 500  # number of radial bins → covers 0 to 50 μm
NZ = 3000  # number of depth bins → covers 0 to 3000 μm


def LaunchPhoton_fig3(z_f, omega0, z0):
    assert z_f > 0
    zf_hat = z_f / z0

    xi1 = max(random.random(), 1e-100)
    r = (
        omega0
        * math.sqrt(1.0 + zf_hat**2)
        * math.sqrt(-math.log(xi1) / 2.0)
    )

    xi2 = random.random()
    theta = 2.0 * math.pi * xi2
    x = r * math.cos(theta)
    y = r * math.sin(theta)

    L = math.sqrt(zf_hat**2 * r**2 / (zf_hat**2 + 1.0) + z_f**2)

    xi3 = random.random()
    if xi3 < 0.5:
        ux = -zf_hat * (zf_hat * x + y) / ((zf_hat**2 + 1.0) * L)
        uy = zf_hat * (x - zf_hat * y) / ((zf_hat**2 + 1.0) * L)
    else:
        ux = -zf_hat * (zf_hat * x - y) / ((zf_hat**2 + 1.0) * L)
        uy = -zf_hat * (x + zf_hat * y) / ((zf_hat**2 + 1.0) * L)

    uz = z_f / L
    assert abs(ux**2 + uy**2 + uz**2 - 1.0) < 1e-10

    return {
        "x": x,
        "y": y,
        "z": 0.0,
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "w": 1.0,
        "dead": False,
        "s": 0.0,
        "sleft": 0.0,
        "n_scatter": 0,
        "has_crossed_focal": False,
    }


def Drop_2d(photon, mu_a, mu_t, fluence_grid, dr, dz):
    dw = photon["w"] * mu_a / mu_t
    photon["w"] -= dw

    r = math.sqrt(photon["x"] ** 2 + photon["y"] ** 2)
    ir = int(r / dr)
    iz = int(photon["z"] / dz)

    if 0 <= ir < fluence_grid.shape[0] and 0 <= iz < fluence_grid.shape[1]:
        fluence_grid[ir, iz] += dw


def Spin_counted_safe(photon, g):
    if g >= 1.0 - 1e-9:
        photon["n_scatter"] += 1
        return
    Spin(photon, g)
    photon["n_scatter"] += 1


def main_photon_loop_fig3(
    photon,
    n_tissue,
    n_ambient,
    mu_a,
    mu_s,
    g,
    fluence_grid,
    dr,
    dz,
    z_f,
    z_max,
    focal_r_list,
    focal_n_list,
    scratch,
):
    mu_t = mu_a + mu_s

    while not photon["dead"]:
        if photon["sleft"] == 0.0:
            photon["s"] = StepSizeInTissue()
        else:
            photon["s"] = photon["sleft"]
            photon["sleft"] = 0.0
        step_physical = photon["s"] / mu_t

        d_top = 1e30
        d_bottom = 1e30
        d_focal = 1e30

        if photon["uz"] < 0.0 and photon["z"] > 0.0:
            d_top = photon["z"] / abs(photon["uz"])

        if photon["uz"] > 0.0:
            d_bottom = (z_max - photon["z"]) / photon["uz"]

        if (
            not photon["has_crossed_focal"]
            and photon["uz"] > 0.0
            and photon["z"] < z_f
        ):
            d_focal = (z_f - photon["z"]) / photon["uz"]

        hit_top = step_physical > d_top
        hit_bottom = step_physical > d_bottom
        hit_focal = step_physical > d_focal

        d_min = min(
            d_top if hit_top else 1e30,
            d_bottom if hit_bottom else 1e30,
            d_focal if hit_focal else 1e30,
        )

        if hit_top and d_min == d_top:
            Hop(photon, d_top)
            photon["sleft"] = (step_physical - d_top) * mu_t
            CrossUpOrNot(photon, n_tissue, n_ambient, scratch)

        elif hit_bottom and d_min == d_bottom:
            Hop(photon, d_bottom)
            photon["dead"] = True

        elif hit_focal and d_min == d_focal:
            Hop(photon, d_focal)
            r_focal = math.sqrt(photon["x"] ** 2 + photon["y"] ** 2)
            focal_r_list.append(r_focal)
            focal_n_list.append(photon["n_scatter"])
            photon["has_crossed_focal"] = True
            photon["sleft"] = (step_physical - d_focal) * mu_t

        else:
            Hop(photon, step_physical)
            Drop_2d(photon, mu_a, mu_t, fluence_grid, dr, dz)
            if not photon["dead"]:
                Spin_counted_safe(photon, g)
                Roulette(photon)


def run_simulation_fig3(
    z_f,
    N_photons,
    omega0=OMEGA0,
    z0=Z0,
    mu_a=MU_A,
    mu_s=MU_S,
    g=G,
    n=N,
    dr=DR,
    dz=DZ,
    nr=NR,
    nz=NZ,
):
    assert z_f > 0, "z_f must be positive"
    n_ambient = n
    n_tissue = n
    z_max = nz * dz

    fluence_grid = numpy.zeros((nr, nz), dtype=float)
    focal_r_list = []
    focal_n_list = []
    scratch = {"total_reflectance": 0.0}

    for _ in range(N_photons):
        photon = LaunchPhoton_fig3(z_f, omega0, z0)
        main_photon_loop_fig3(
            photon,
            n_tissue,
            n_ambient,
            mu_a,
            mu_s,
            g,
            fluence_grid,
            dr,
            dz,
            z_f,
            z_max,
            focal_r_list,
            focal_n_list,
            scratch,
        )

    for ir in range(nr):
        area = math.pi * dr**2 * (2 * ir + 1)
        fluence_grid[ir, :] /= area * N_photons

    return {
        "fluence_grid": fluence_grid,
        "focal_r_list": focal_r_list,
        "focal_n_list": focal_n_list,
        "z_f": z_f,
        "dr": dr,
        "dz": dz,
    }


# Baseline (no angular deflection): call run_simulation_fig3(..., g=1.0) with default mu_s=MU_S;
# do not set mu_s=0 (that makes mu_t=mu_a and removes all weight on the first Drop_2d).
N_PHOTONS = 1_000_000  # paper uses 2×10¹⁰; 10⁶ is usable for development


if __name__ == "__main__":
    all_results = {}
    for zf_lt in ZF_OVER_LT:
        z_f = zf_lt * L_T_PRIME
        print(f"Running z_f/l_t' = {zf_lt}  (z_f = {z_f * 1e4:.1f} μm) ...")
        all_results[zf_lt] = run_simulation_fig3(z_f, N_PHOTONS)

    all_results["baseline"] = run_simulation_fig3(
        z_f=0.1 * L_T_PRIME, N_photons=N_PHOTONS, g=1.0
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    ax = axes[0]
    for zf_lt, res in all_results.items():
        iz_f = int(res["z_f"] / DZ)
        assert iz_f < res["fluence_grid"].shape[1], (
            f"iz_f={iz_f} out of range; increase NZ"
        )
        psf = res["fluence_grid"][:, iz_f]
        peak = psf.max()
        if peak > 0:
            psf = psf / peak
        r_um = numpy.arange(NR) * DR * 1e4
        r_lat = numpy.concatenate([-r_um[::-1], r_um[1:]])
        p_lat = numpy.concatenate([psf[::-1], psf[1:]])
        label = "baseline" if zf_lt == "baseline" else f"z_f/l_t'={zf_lt}"
        ax.plot(r_lat, p_lat, label=label)
    ax.set_xlabel("Lateral position (μm)")
    ax.set_ylabel("Normalized fluence")
    ax.set_xlim(-50, 50)
    ax.set_title("(a) PSF at focal plane")
    ax.legend(fontsize=6)

    ax = axes[1]
    for zf_lt, res in all_results.items():
        iz_f = int(res["z_f"] / DZ)
        assert iz_f < res["fluence_grid"].shape[1], (
            f"iz_f={iz_f} out of range; increase NZ"
        )
        psf = res["fluence_grid"][:, iz_f]
        peak = psf.max()
        if peak > 0:
            psf = psf / peak
        r_um = numpy.arange(NR) * DR * 1e4
        r_lat = numpy.concatenate([-r_um[::-1], r_um[1:]])
        p_lat = numpy.concatenate([psf[::-1], psf[1:]])
        label = "baseline" if zf_lt == "baseline" else f"z_f/l_t'={zf_lt}"
        ax.plot(r_lat, p_lat, label=label)
    ax.set_xlabel("Lateral position (μm)")
    ax.set_ylabel("Normalized fluence")
    ax.set_xlim(-5, 5)
    ax.set_title("(b) PSF close-up")
    ax.legend(fontsize=6)

    ax = axes[2]
    R_BINS_UM = numpy.linspace(0, 4, 20)
    r_centers = 0.5 * (R_BINS_UM[:-1] + R_BINS_UM[1:])
    for zf_lt, res in all_results.items():
        if zf_lt == "baseline":
            continue
        r_arr = numpy.array(res["focal_r_list"]) * 1e4
        n_arr = numpy.array(res["focal_n_list"])
        mean_ns = []
        for i in range(len(R_BINS_UM) - 1):
            mask = (r_arr >= R_BINS_UM[i]) & (r_arr < R_BINS_UM[i + 1])
            mean_ns.append(n_arr[mask].mean() if mask.any() else 0.0)
        ax.plot(r_centers, mean_ns, label=f"z_f/l_t'={zf_lt}")
    ax.set_xlabel("Radial position (μm)")
    ax.set_ylabel("Mean # scattering events")
    ax.set_title("(c) N_s at focal plane")
    ax.legend(fontsize=6)

    ax = axes[3]
    z_um = numpy.arange(NZ) * DZ * 1e4
    for zf_lt, res in all_results.items():
        on_ax = res["fluence_grid"][0, :]
        label = "baseline" if zf_lt == "baseline" else f"z_f/l_t'={zf_lt}"
        ax.plot(z_um, on_ax, label=label)
    ax.set_xlabel("Depth (μm)")
    ax.set_ylabel("Fluence (a.u.)")
    ax.set_title("(d) On-axis fluence")
    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig("fig3_reproduction.png", dpi=150)
    plt.show()
