import math
import multiprocessing
import random
import statistics

from scipy import stats

from simulation_runnable_partb import (
    CrossUpOrNot,
    Hop,
    LaunchPhoton,
    Roulette,
    Rspecular,
    Spin,
    run_simulation,
)


def StepSizeInTissue_b(mu_s):
    """Sample the physical distance to the next scattering event."""
    return -math.log(max(random.random(), 1e-100)) / mu_s


def Drop_b(photon, mu_a, s_physical, results):
    """Score exact absorption over a full physical step using Beer-Lambert."""
    dw = photon["w"] * (1.0 - math.exp(-mu_a * s_physical))
    photon["w"] -= dw

    iz = int(photon["z"] / results["dz"])
    if 0 <= iz < len(results["fluence"]):
        results["fluence"][iz] += dw


def main_photon_loop_b(photon, n_tissue, n_ambient, mu_a, mu_s, g, results):
    """Track one photon using the scattering-step formulation."""
    while not photon["dead"]:
        if photon["sleft"] == 0.0:
            step_physical = StepSizeInTissue_b(mu_s)
        else:
            step_physical = photon["sleft"]
            photon["sleft"] = 0.0

        hit_boundary = False
        d_boundary = 0.0

        if photon["uz"] < 0.0:
            d_boundary = photon["z"] / abs(photon["uz"])
            if step_physical > d_boundary:
                hit_boundary = True

        if hit_boundary:
            Drop_b(photon, mu_a, d_boundary, results)
            Hop(photon, d_boundary)
            photon["sleft"] = step_physical - d_boundary
            CrossUpOrNot(photon, n_tissue, n_ambient, results)
        else:
            Hop(photon, step_physical)
            Drop_b(photon, mu_a, step_physical, results)

            if not photon["dead"]:
                Spin(photon, g)
                Roulette(photon)


def run_simulation_b(n_rel, mu_a, mu_s, g, N_photons, dz=0.01, nz=500):
    """Run the semi-infinite Monte Carlo simulation for algorithm (b)."""
    n_ambient = 1.0
    n_tissue = n_rel * n_ambient

    results = {
        "total_reflectance": 0.0,
        "fluence": [0.0] * nz,
        "dz": dz,
    }

    for _ in range(N_photons):
        photon = LaunchPhoton(n_ambient, n_tissue)
        main_photon_loop_b(photon, n_tissue, n_ambient, mu_a, mu_s, g, results)

    norm = dz * N_photons
    results["fluence"] = [value / norm for value in results["fluence"]]
    return results


def _single_run(args):
    run_fn, kwargs, photons_per_run = args
    return run_fn(**kwargs, N_photons=photons_per_run)


def collect_stats(run_fn, n_runs, photons_per_run, **kwargs):
    """Run a simulation repeatedly in parallel and collect per-run statistics."""
    if "N_photons" in kwargs:
        raise ValueError("Pass photons_per_run separately; do not include N_photons in kwargs.")

    specular = Rspecular(1.0, kwargs["n_rel"])

    with multiprocessing.Pool() as pool:
        run_results = pool.map(
            _single_run,
            [(run_fn, kwargs, photons_per_run)] * n_runs,
        )

    diffuse_reflectances = []
    absorptions = []
    for result in run_results:
        diffuse_r = result["total_reflectance"] / photons_per_run
        absorption = 1.0 - specular - diffuse_r
        diffuse_reflectances.append(diffuse_r)
        absorptions.append(absorption)

    return diffuse_reflectances, absorptions


def run_comparison():
    n_rel = 1.0
    mu_a = 0.1
    mu_s = 100.0
    g = 0.9
    n_runs = 30
    photons_per_run = 10_000

    R_a, A_a = collect_stats(
        run_simulation,
        n_runs,
        photons_per_run,
        n_rel=n_rel,
        mu_a=mu_a,
        mu_s=mu_s,
        g=g,
    )
    R_b, A_b = collect_stats(
        run_simulation_b,
        n_runs,
        photons_per_run,
        n_rel=n_rel,
        mu_a=mu_a,
        mu_s=mu_s,
        g=g,
    )

    t_r, p_r = stats.ttest_ind(R_a, R_b)
    t_a, p_a = stats.ttest_ind(A_a, A_b)

    print("=== Problem 11: Algorithm Comparison ===")
    print(f"Parameters: n_rel={n_rel}, mu_a={mu_a}, mu_s={mu_s}, g={g}")
    print(f"Runs: {n_runs} x {photons_per_run} photons each")
    print()
    print("--- Diffuse Reflectance ---")
    print(
        f"Algorithm A mean: {statistics.mean(R_a):.6f}  std: {statistics.stdev(R_a):.6f}"
    )
    print(
        f"Algorithm B mean: {statistics.mean(R_b):.6f}  std: {statistics.stdev(R_b):.6f}"
    )
    print(f"t-statistic: {t_r:.6f}   p-value: {p_r:.6f}")
    print(
        "Result: STATISTICALLY IDENTICAL (p > 0.05)"
        if p_r > 0.05
        else "Result: NOT identical (p <= 0.05)"
    )
    print()
    print("--- Total Absorption ---")
    print(
        f"Algorithm A mean: {statistics.mean(A_a):.6f}  std: {statistics.stdev(A_a):.6f}"
    )
    print(
        f"Algorithm B mean: {statistics.mean(A_b):.6f}  std: {statistics.stdev(A_b):.6f}"
    )
    print(f"t-statistic: {t_a:.6f}   p-value: {p_a:.6f}")
    print(
        "Result: STATISTICALLY IDENTICAL (p > 0.05)"
        if p_a > 0.05
        else "Result: NOT identical (p <= 0.05)"
    )


if __name__ == "__main__":
    run_comparison()
