import math
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress expected numpy warnings for empty bin operations in log scale
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Constants & Optical Properties ---
THRESHOLD = 1e-4
CHANCE = 10
COSZERO = 1.0 - 1e-12

# Tissue properties
g = 0.9
mu_s = 0.01  # 100 cm^-1 = 0.01 um^-1
mu_a = 1e-5  # 0.1 cm^-1 = 1e-5 um^-1
mu_t = mu_a + mu_s

# Optical system parameters
wavelength = 0.570 # um
NA = 0.1
w0 = (1.0 / math.pi) * (wavelength / NA) 
z0 = (math.pi * w0**2) / wavelength      

# Grid properties
dr = 0.1 # um
dz = 1.0 # um
lt_prime = 1.0 / (mu_s * (1.0 - g) + mu_a) # Transport mean free path (~990 um)
Nr = 500  
Nz = 3500 

def LaunchHyperboloidPhoton(zf):
    """Initialize photon using the hyperboloid focusing method."""
    z_hat_f = zf / z0

    xi1 = max(random.random(), 1e-100)
    r = w0 * math.sqrt(1.0 + z_hat_f**2) * math.sqrt(-math.log(xi1) / 2.0)

    xi2 = random.random()
    theta = 2.0 * math.pi * xi2
    x = r * math.cos(theta)
    y = r * math.sin(theta)

    xi3 = random.random()
    L = math.sqrt((z_hat_f**2 * r**2) / (z_hat_f**2 + 1.0) + zf**2)

    if xi3 < 0.5: 
        ux = -z_hat_f * (z_hat_f * x + y) / ((z_hat_f**2 + 1.0) * L)
        uy = z_hat_f * (x - z_hat_f * y) / ((z_hat_f**2 + 1.0) * L)
    else:         
        ux = -z_hat_f * (z_hat_f * x - y) / ((z_hat_f**2 + 1.0) * L)
        uy = -z_hat_f * (x + z_hat_f * y) / ((z_hat_f**2 + 1.0) * L)
    uz = zf / L

    return {
        "x": x, "y": y, "z": 0.0,
        "ux": ux, "uy": uy, "uz": uz,
        "w": 1.0, 
        "dead": False,
        "scatters": 0
    }

def StepSizeInTissue():
    return -math.log(max(random.random(), 1e-100))

def Spin(photon):
    photon["scatters"] += 1
    if g == 0.0:
        cos_theta = 2.0 * random.random() - 1.0
    else:
        temp = (1.0 - g*g) / (1.0 - g + 2.0 * g * random.random())
        cos_theta = (1.0 + g*g - temp*temp) / (2.0 * g)

    cos_theta = max(-1.0, min(1.0, cos_theta))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta*cos_theta))
    psi = 2.0 * math.pi * random.random()
    cos_psi = math.cos(psi)
    sin_psi = math.sin(psi)

    ux, uy, uz = photon["ux"], photon["uy"], photon["uz"]

    if abs(uz) > COSZERO:
        photon["ux"] = sin_theta * cos_psi
        photon["uy"] = sin_theta * sin_psi
        photon["uz"] = cos_theta * (1.0 if uz > 0.0 else -1.0)
    else:
        temp = math.sqrt(1.0 - uz*uz)
        photon["ux"] = (sin_theta * (ux * uz * cos_psi - uy * sin_psi)) / temp + ux * cos_theta
        photon["uy"] = (sin_theta * (uy * uz * cos_psi + ux * sin_psi)) / temp + uy * cos_theta
        photon["uz"] = -sin_theta * cos_psi * temp + uz * cos_theta

def Roulette(photon):
    if photon["w"] < THRESHOLD:
        if random.random() <= 1.0 / CHANCE:
            photon["w"] *= CHANCE
        else:
            photon["w"] = 0.0
            photon["dead"] = True

def run_focused_simulation(zf, N_photons):
    fluence_grid = np.zeros((Nr, Nz))
    scatter_sum = np.zeros((Nr, Nz))
    scatter_weight = np.zeros((Nr, Nz))

    print_interval = max(1, N_photons // 10)

    for i in range(N_photons):
        if (i + 1) % print_interval == 0:
            print(f"  ...processed {i + 1:,} / {N_photons:,} packets ({(i + 1) / N_photons * 100:.0f}%)")

        photon = LaunchHyperboloidPhoton(zf)

        while not photon["dead"]:
            s = StepSizeInTissue()
            step_physical = s / mu_t

            # Exact Boundary Escape Logic (Index Matched)
            if photon["uz"] < 0.0:
                d_boundary = photon["z"] / abs(photon["uz"])
                if step_physical > d_boundary:
                    photon["x"] += d_boundary * photon["ux"]
                    photon["y"] += d_boundary * photon["uy"]
                    photon["z"] = 0.0
                    photon["dead"] = True
                    continue

            photon["x"] += step_physical * photon["ux"]
            photon["y"] += step_physical * photon["uy"]
            photon["z"] += step_physical * photon["uz"]

            dw = photon["w"] * (mu_a / mu_t)
            photon["w"] -= dw

            ir = int(math.sqrt(photon["x"]**2 + photon["y"]**2) / dr)
            iz = int(photon["z"] / dz)

            if 0 <= ir < Nr and 0 <= iz < Nz:
                fluence_grid[ir, iz] += dw
                scatter_sum[ir, iz] += dw * photon["scatters"]
                scatter_weight[ir, iz] += dw

            if photon["w"] <= 0.0:
                photon["dead"] = True
            else:
                Spin(photon)
                Roulette(photon)

    # Volumetric Normalization
    for ir in range(Nr):
        V = 2.0 * math.pi * (ir + 0.5) * (dr**2) * dz
        fluence_grid[ir, :] /= (mu_a * V * N_photons)

    scatters_avg = np.divide(scatter_sum, scatter_weight, out=np.zeros_like(scatter_sum), where=scatter_weight!=0)

    return fluence_grid, scatters_avg

# --- Execution & Plotting ---
N_PACKETS = 10000 
zf_values = [0.1, 1.1]

results = {}
for zf_lt in zf_values:
    print(f"\n--- Starting simulation for focal depth zf = {zf_lt} lt' ---")
    zf_um = zf_lt * lt_prime
    F, S = run_focused_simulation(zf_um, N_PACKETS)
    results[zf_lt] = {"F": F, "S": S, "iz_f": int(zf_um / dz)}

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Optimal Radial Coordinates
ir_indices = np.arange(Nr)
r_arr = ((ir_indices + 0.5) + (1.0 / (12.0 * (ir_indices + 0.5)))) * dr
x_arr = np.concatenate((-r_arr[::-1], r_arr))
z_axis_lt = ((np.arange(Nz) + 0.5) * dz) / lt_prime

ax = axs[0, 0]
ax.set_title("(a) Illumination PSF on focal plane")
ax.set_xlabel("Lateral position [μm]")
ax.set_ylabel("Normalized fluence")
ax.set_xlim(-50, 50)

ax_b = axs[0, 1]
ax_b.set_title("(b) Close-up of PSF")
ax_b.set_xlabel("Lateral position [μm]")
ax_b.set_ylabel("Normalized fluence")
ax_b.set_xlim(-5, 5)

ax_c = axs[1, 0]
ax_c.set_title("(c) Scattering events on focal plane")
ax_c.set_xlabel("Radial position [μm]")
ax_c.set_ylabel("Number of scattering events")
ax_c.set_xlim(0, 5)

ax_d = axs[1, 1]
ax_d.set_title("(d) On-axis fluence distributions")
ax_d.set_xlabel("Depth from surface [lt']")
ax_d.set_ylabel("On-axis fluence [a.u.]")
ax_d.set_yscale('log')
ax_d.set_xlim(0, 1.9)

# Analytical Baseline for No Scattering (Gaussian Intensity Profile)
F_baseline = np.exp(-2.0 * (r_arr**2) / (w0**2))
F_sym_base = np.concatenate((F_baseline[::-1], F_baseline)) 

ax.plot(x_arr, F_sym_base, 'k--', label="No scattering")
ax_b.plot(x_arr, F_sym_base, 'k--', label="No scattering")

colors = {0.1: 'blue', 1.1: 'brown'}

for zf_lt in zf_values:
    data = results[zf_lt]
    F_focal = data["F"][:, data["iz_f"]]
    
    max_val = np.max(F_focal) if np.max(F_focal) > 0 else 1
    F_sym = np.concatenate((F_focal[::-1], F_focal)) / max_val

    c = colors[zf_lt]
    ax.plot(x_arr, F_sym, color=c, label=f"zf = {zf_lt} lt'")
    ax_b.plot(x_arr, F_sym, color=c, label=f"zf = {zf_lt} lt'")

    ax_c.plot(r_arr, data["S"][:, data["iz_f"]], color=c, label=f"zf = {zf_lt} lt'")
    
    on_axis = np.copy(data["F"][0, :])
    on_axis[on_axis <= 0] = np.nan 
    ax_d.plot(z_axis_lt, on_axis, color=c, label=f"zf = {zf_lt} lt'")

ax.legend(fontsize=8)
ax_b.legend(fontsize=8)
ax_c.legend(fontsize=8)
ax_d.legend(fontsize=8)

plt.tight_layout()
plt.show()