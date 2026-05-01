import math
import random
import os
import multiprocessing as mp
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
        "s": 0.0,
        "sleft": 0.0,
        "n_scatter": 0,
        "has_crossed_focal": False,
    }

def StepSizeInTissue():
    return -math.log(max(random.random(), 1e-100))

def Hop(photon, s):
    photon["x"] += s * photon["ux"]
    photon["y"] += s * photon["uy"]
    photon["z"] += s * photon["uz"]

def Spin(photon, g_val):
    if g_val == 0.0:
        cos_theta = 2.0 * random.random() - 1.0
    else:
        temp = (1.0 - g_val*g_val) / (1.0 - g_val + 2.0 * g_val * random.random())
        cos_theta = (1.0 + g_val*g_val - temp*temp) / (2.0 * g_val)

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

def Spin_counted_safe(photon, g_val):
    if g_val >= 1.0 - 1e-9:
        photon["n_scatter"] += 1
        return
    Spin(photon, g_val)
    photon["n_scatter"] += 1

def Drop_2d(photon, mu_a_val, mu_t_val, fluence_grid, dr_val, dz_val):
    dw = photon["w"] * (mu_a_val / mu_t_val)
    photon["w"] -= dw

    r = math.sqrt(photon["x"] ** 2 + photon["y"] ** 2)
    ir = int(r / dr_val)
    iz = int(photon["z"] / dz_val)

    if 0 <= ir < fluence_grid.shape[0] and 0 <= iz < fluence_grid.shape[1]:
        fluence_grid[ir, iz] += dw

def CrossUpOrNot(photon, n_tissue, n_ambient, scratch):
    photon["dead"] = True
    scratch["total_reflectance"] += photon["w"]
    photon["w"] = 0.0

def Roulette(photon):
    if photon["w"] < THRESHOLD:
        if random.random() <= 1.0 / CHANCE:
            photon["w"] *= CHANCE
        else:
            photon["w"] = 0.0
            photon["dead"] = True

def main_photon_loop_fig3(
    photon,
    n_tissue,
    n_ambient,
    mu_a_val,
    mu_s_val,
    g_val,
    fluence_grid,
    dr_val,
    dz_val,
    z_f,
    z_max,
    focal_r_list,
    focal_n_list,
    scratch,
):
    mu_t_val = mu_a_val + mu_s_val

    while not photon["dead"]:
        if photon["sleft"] == 0.0:
            photon["s"] = StepSizeInTissue()
        else:
            photon["s"] = photon["sleft"]
            photon["sleft"] = 0.0
        step_physical = photon["s"] / mu_t_val

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
            photon["sleft"] = (step_physical - d_top) * mu_t_val
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
            photon["sleft"] = (step_physical - d_focal) * mu_t_val

        else:
            Hop(photon, step_physical)
            Drop_2d(photon, mu_a_val, mu_t_val, fluence_grid, dr_val, dz_val)
            if not photon["dead"]:
                Spin_counted_safe(photon, g_val)
                Roulette(photon)

_shared_counter = None
_total_photons = None

def init_worker(counter, total):
    global _shared_counter, _total_photons
    _shared_counter = counter
    _total_photons = total

def _worker_simulation(args):
    """Worker function for parallel processing."""
    N_chunk, zf, simulate_scattering = args
    
    # Ensure completely independent random streams across processes
    random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    fluence_grid = np.zeros((Nr, Nz))
    focal_r_list = []
    focal_n_list = []
    scratch = {"total_reflectance": 0.0}
    local_mu_s = mu_s if simulate_scattering else 0.0

    global _shared_counter, _total_photons
    local_count = 0
    batch_size = max(1, N_chunk // 100)
    progress_interval = max(1, _total_photons // 10)

    for i in range(N_chunk):
        photon = LaunchHyperboloidPhoton(zf)
        main_photon_loop_fig3(
            photon, 
            n_tissue=1.33, 
            n_ambient=1.33, 
            mu_a_val=mu_a, 
            mu_s_val=local_mu_s, 
            g_val=g, 
            fluence_grid=fluence_grid, 
            dr_val=dr, 
            dz_val=dz, 
            z_f=zf, 
            z_max=Nz * dz, 
            focal_r_list=focal_r_list, 
            focal_n_list=focal_n_list, 
            scratch=scratch
        )

        local_count += 1
        if local_count >= batch_size or i == N_chunk - 1:
            with _shared_counter.get_lock():
                old_val = _shared_counter.value
                _shared_counter.value += local_count
                new_val = _shared_counter.value
            
            old_step = old_val // progress_interval
            new_step = new_val // progress_interval
            
            if new_step > old_step:
                for step in range(old_step + 1, new_step + 1):
                    cross = step * progress_interval
                    if cross <= _total_photons:
                        percent = int((cross / _total_photons) * 100)
                        print(f"  ...processed {cross:,} / {_total_photons:,} packets ({percent}%)")
            
            local_count = 0

    return fluence_grid, focal_r_list, focal_n_list

def run_focused_simulation(zf, N_photons, simulate_scattering=True):
    """Distributes the simulation across all available CPU cores."""
    num_cores = mp.cpu_count()
    print(f"  ...launching {num_cores} parallel workers to track {N_photons:,} packets...")

    # Divide the workload among the CPU cores
    chunk_size = N_photons // num_cores
    chunks = [chunk_size] * num_cores
    chunks[-1] += N_photons % num_cores  # Add any remainder to the last core
    
    args = [(chunk, zf, simulate_scattering) for chunk in chunks]

    total_fluence_grid = np.zeros((Nr, Nz))
    total_focal_r_list = []
    total_focal_n_list = []

    # Execute parallel pool
    counter = mp.Value('i', 0)
    with mp.Pool(processes=num_cores, initializer=init_worker, initargs=(counter, N_photons)) as pool:
        results = pool.map(_worker_simulation, args)

    # Aggregate results from all workers
    for res_grid, res_r_list, res_n_list in results:
        total_fluence_grid += res_grid
        total_focal_r_list.extend(res_r_list)
        total_focal_n_list.extend(res_n_list)

    # Volumetric Normalization
    for ir in range(Nr):
        V = 2.0 * math.pi * (ir + 0.5) * (dr**2) * dz
        total_fluence_grid[ir, :] /= (mu_a * V * N_photons)

    return total_fluence_grid, total_focal_r_list, total_focal_n_list

# ====================================================================
# Execution block must be protected under __main__ for multiprocessing
# ====================================================================
if __name__ == '__main__':
    # --- Execution & Plotting ---
    N_PACKETS = 1000000
    # Using a single focal depth (0.1)
    zf_values = [1.1] 

    results = {}
    for zf_lt in zf_values:
        print(f"\n--- Tracking packets for focal depth zf = {zf_lt} lt' ---")
        zf_um = zf_lt * lt_prime
        F, R_list, N_list = run_focused_simulation(zf_um, N_PACKETS)
        results[zf_lt] = {"F": F, "R_list": R_list, "N_list": N_list, "iz_f": int(zf_um / dz)}

    # print("\n--- Computing analytical no-scattering baseline ---")
    # F_noscatter, _, _ = run_focused_simulation(1.0 * lt_prime, N_PACKETS, simulate_scattering=False)

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

    #baseline_iz = int(1.0 * lt_prime / dz)
    #F_baseline = F_noscatter[:, baseline_iz]
    #max_baseline = np.max(F_baseline) if np.max(F_baseline) > 0 else 1
    #F_sym_base = np.concatenate((F_baseline[::-1], F_baseline)) / max_baseline
    
    #ax.plot(x_arr, F_sym_base, 'k--', label="No scattering")
    #ax_b.plot(x_arr, F_sym_base, 'k--', label="No scattering")

    # Setup for Plot (c) exact binning logic
    R_BINS_UM = np.linspace(0, 5, 26) # Bin boundaries from 0 to 5 um
    r_centers = 0.5 * (R_BINS_UM[:-1] + R_BINS_UM[1:])

    desktop_path = os.path.expanduser("~/Desktop")

    for zf_lt in zf_values:
        data = results[zf_lt]
        F_focal = data["F"][:, data["iz_f"]]
        
        max_val = np.max(F_focal) if np.max(F_focal) > 0 else 1
        F_sym = np.concatenate((F_focal[::-1], F_focal)) / max_val

        ax.plot(x_arr, F_sym, label=f"zf = {zf_lt} lt'")
        ax_b.plot(x_arr, F_sym, label=f"zf = {zf_lt} lt'")

        # --- Data saving structure ---
        print(f"\nSaving CSV data files to Desktop for zf = {zf_lt} lt'...")
        
        # 1. Save PSF Profile (Lateral Position vs. Fluence)
        psf_data = np.column_stack((x_arr, F_sym))
        np.savetxt(os.path.join(desktop_path, f"PSF_data_zf_{zf_lt}.csv"), psf_data, delimiter=",", header="Lateral_Position_um,Normalized_Fluence", comments="")

        # Plot and save Scattering Events
        if zf_lt <= 1.7:
            f_r_arr = np.array(data["R_list"])
            f_n_arr = np.array(data["N_list"])
            mean_ns = []
            for i in range(len(R_BINS_UM) - 1):
                mask = (f_r_arr >= R_BINS_UM[i]) & (f_r_arr < R_BINS_UM[i + 1])
                mean_ns.append(f_n_arr[mask].mean() if mask.any() else 0.0)
                
            ax_c.plot(r_centers, mean_ns, label=f"zf = {zf_lt} lt'")
            
            # 2. Save Scattering Events Profile
            scatter_data = np.column_stack((r_centers, mean_ns))
            np.savetxt(os.path.join(desktop_path, f"Scattering_data_zf_{zf_lt}.csv"), scatter_data, delimiter=",", header="Radial_Position_um,Mean_Scatters", comments="")
            
            # Plot and save On-axis Fluence
            on_axis = np.copy(data["F"][0, :])
            on_axis[on_axis <= 0] = np.nan 
            ax_d.plot(z_axis_lt, on_axis, label=f"zf = {zf_lt} lt'")
            
            # 3. Save On-axis Fluence Profile
            onaxis_data = np.column_stack((z_axis_lt, on_axis))
            np.savetxt(os.path.join(desktop_path, f"OnAxis_data_zf_{zf_lt}.csv"), onaxis_data, delimiter=",", header="Depth_lt,OnAxis_Fluence", comments="")

    # 4. Optional: Save Baseline No-Scattering PSF for reference
    #baseline_data = np.column_stack((x_arr, F_sym_base))
    #np.savetxt(os.path.join(desktop_path, "PSF_Baseline_NoScattering.csv"), baseline_data, delimiter=",", header="Lateral_Position_um,Normalized_Fluence", comments="")
    #print("Baseline CSV saved.")

    ax.legend(fontsize=8)
    ax_b.legend(fontsize=8)
    ax_c.legend(fontsize=8)
    ax_d.legend(fontsize=8)

    plt.tight_layout()
    plt.show()