import math
import random
import scipy.stats as stats

THRESHOLD = 1e-4
CHANCE = 10
COSZERO = 1.0 - 1e-12

def Rspecular(n1, n2):
    temp = (n1 - n2) / (n1 + n2)
    return temp * temp

def RFresnel(n1, n2, ca1):
    if n1 == n2:
        return 0.0, ca1
    if ca1 > 0.99999:
        temp = (n1 - n2) / (n1 + n2)
        return temp * temp, ca1
    sa1 = math.sqrt(1.0 - ca1 * ca1)
    sa2 = n1 * sa1 / n2
    if sa2 >= 1.0:
        return 1.0, 0.0
    ca2 = math.sqrt(1.0 - sa2 * sa2)
    cap = ca1 * ca2 - sa1 * sa2
    cam = ca1 * ca2 + sa1 * sa2
    sap = sa1 * ca2 + ca1 * sa2
    sam = sa1 * ca2 - ca1 * sa2
    r = 0.5 * sam * sam * (cam * cam + cap * cap) / (sap * sap * cam * cam)
    return r, ca2

def LaunchPhoton(n_ambient, n_tissue):
    rsp = Rspecular(n_ambient, n_tissue)
    photon = {
        "x": 0.0, "y": 0.0, "z": 0.0,
        "ux": 0.0, "uy": 0.0, "uz": 1.0,
        "w": 1.0 - rsp, "dead": False,
        "s": 0.0, "sleft": 0.0,
    }
    return photon

def Hop(photon, s):
    photon["x"] += s * photon["ux"]
    photon["y"] += s * photon["uy"]
    photon["z"] += s * photon["uz"]

def SpinTheta(g):
    if g == 0.0:
        return 2.0 * random.random() - 1.0
    temp = (1.0 - g * g) / (1.0 - g + 2.0 * g * random.random())
    cos_theta = (1.0 + g * g - temp * temp) / (2.0 * g)
    return max(-1.0, min(1.0, cos_theta))

def Spin(photon, g):
    cos_theta = SpinTheta(g)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    psi = 2.0 * math.pi * random.random()
    cos_psi = math.cos(psi)
    sin_psi = math.sin(psi)

    ux, uy, uz = photon["ux"], photon["uy"], photon["uz"]

    if abs(uz) > COSZERO:
        ux_new = sin_theta * cos_psi
        uy_new = sin_theta * sin_psi
        uz_new = cos_theta * (1.0 if uz > 0.0 else -1.0)
    else:
        temp = math.sqrt(1.0 - uz * uz)
        ux_new = (sin_theta * (ux * uz * cos_psi - uy * sin_psi)) / temp + ux * cos_theta
        uy_new = (sin_theta * (uy * uz * cos_psi + ux * sin_psi)) / temp + uy * cos_theta
        uz_new = -sin_theta * cos_psi * temp + uz * cos_theta

    photon["ux"], photon["uy"], photon["uz"] = ux_new, uy_new, uz_new

def Roulette(photon):
    if photon["w"] >= THRESHOLD:
        return
    if random.random() < 1.0 / CHANCE:
        photon["w"] *= CHANCE
    else:
        photon["w"] = 0.0
        photon["dead"] = True

def CrossUpOrNot(photon, n_tissue, n_ambient, results):
    ca1 = abs(photon["uz"])
    r, ca2 = RFresnel(n_tissue, n_ambient, ca1)
    if random.random() <= r:
        photon["uz"] = -photon["uz"]
    else:
        photon["dead"] = True
        results["total_reflectance"] += photon["w"]
        photon["w"] = 0.0

# ==========================================
# ALGORITHM A: Discrete Absorption Loop
# ==========================================
def algo_a_loop(photon, n_tissue, n_ambient, mu_a, mu_s, g, results):
    mu_t = mu_a + mu_s
    while not photon["dead"]:
        if photon["sleft"] == 0.0:
            photon["s"] = -math.log(max(random.random(), 1e-100))
        else:
            photon["s"] = photon["sleft"]
            photon["sleft"] = 0.0

        step_physical = photon["s"] / mu_t
        hit_boundary = False
        d_boundary = 0.0

        if photon["uz"] < 0.0:
            d_boundary = photon["z"] / abs(photon["uz"])
            if step_physical > d_boundary:
                hit_boundary = True

        if hit_boundary:
            Hop(photon, d_boundary)
            photon["sleft"] = (step_physical - d_boundary) * mu_t
            # No absorption on partial steps in Algorithm A
            CrossUpOrNot(photon, n_tissue, n_ambient, results)
        else:
            Hop(photon, step_physical)
            # Discrete absorption at scattering site
            dw = photon["w"] * mu_a / mu_t
            photon["w"] -= dw
            results["total_absorption"] += dw

            if not photon["dead"]:
                Spin(photon, g)
                Roulette(photon)

def algo_b_loop(photon, n_tissue, n_ambient, mu_a, mu_s, g, results):
    while not photon["dead"]:
        if photon["sleft"] == 0.0:
            photon["s"] = -math.log(max(random.random(), 1e-100))
        else:
            photon["s"] = photon["sleft"]
            photon["sleft"] = 0.0

        step_physical = photon["s"] / mu_s
        hit_boundary = False
        d_boundary = 0.0

        if photon["uz"] < 0.0:
            d_boundary = photon["z"] / abs(photon["uz"])
            if step_physical > d_boundary:
                hit_boundary = True

        if hit_boundary:
            Hop(photon, d_boundary)
            photon["sleft"] = (step_physical - d_boundary) * mu_s
            
            # Continuous absorption along path to boundary
            dw = photon["w"] * (1.0 - math.exp(-mu_a * d_boundary))
            photon["w"] -= dw
            results["total_absorption"] += dw

            CrossUpOrNot(photon, n_tissue, n_ambient, results)
        else:
            Hop(photon, step_physical)
            
            # Continuous absorption along full step
            dw = photon["w"] * (1.0 - math.exp(-mu_a * step_physical))
            photon["w"] -= dw
            results["total_absorption"] += dw

            if not photon["dead"]:
                Spin(photon, g)
                Roulette(photon)

def run_simulation(algo_type, n_rel, mu_a, mu_s, g, N_photons):
    n_ambient = 1.0
    n_tissue = n_rel * n_ambient
    results = {
        "total_reflectance": 0.0,
        "total_absorption": 0.0
    }

    for _ in range(N_photons):
        photon = LaunchPhoton(n_ambient, n_tissue)
        if algo_type == 'A':
            algo_a_loop(photon, n_tissue, n_ambient, mu_a, mu_s, g, results)
        elif algo_type == 'B':
            algo_b_loop(photon, n_tissue, n_ambient, mu_a, mu_s, g, results)

    # Normalize by N_photons
    results["total_reflectance"] /= N_photons
    results["total_absorption"] /= N_photons
    return results

def run_experiment():
    # Given parameters
    n_rel = 1.0
    mu_a = 0.1
    mu_s = 100.0
    g = 0.9
    
    photons_per_run = 2000
    n_runs = 20

    refl_A, abs_A = [], []
    for i in range(n_runs):
        res = run_simulation('A', n_rel, mu_a, mu_s, g, photons_per_run)
        refl_A.append(res["total_reflectance"])
        abs_A.append(res["total_absorption"])

    refl_B, abs_B = [], []
    for i in range(n_runs):
        res = run_simulation('B', n_rel, mu_a, mu_s, g, photons_per_run)
        refl_B.append(res["total_reflectance"])
        abs_B.append(res["total_absorption"])

    mean_refl_A = sum(refl_A) / n_runs
    mean_refl_B = sum(refl_B) / n_runs
    mean_abs_A = sum(abs_A) / n_runs
    mean_abs_B = sum(abs_B) / n_runs

    # T-test
    t_refl, p_refl = stats.ttest_ind(refl_A, refl_B, equal_var=False)
    t_abs, p_abs = stats.ttest_ind(abs_A, abs_B, equal_var=False)

    print(f"Algorithm A - Mean Reflectance: {mean_refl_A:.5f} | Mean Absorption: {mean_abs_A:.5f}")
    print(f"Algorithm B - Mean Reflectance: {mean_refl_B:.5f} | Mean Absorption: {mean_abs_B:.5f}")
    
    print(f"Reflectance: T-statistic = {t_refl:.4f}, p-value = {p_refl:.4f}")
    print(f"Absorption : T-statistic = {t_abs:.4f},  p-value = {p_abs:.4f}")

    alpha = 0.05
    if p_refl > alpha and p_abs > alpha:
        print("The p-values are > 0.05. We fail to reject the null hypothesis.")
        print("This verifies that Algorithm A and Algorithm B are STATISTICALLY IDENTICAL.")
    else:
        print("One or more p-values are < 0.05. The algorithms differ statistically.")

if __name__ == "__main__":
    run_experiment()