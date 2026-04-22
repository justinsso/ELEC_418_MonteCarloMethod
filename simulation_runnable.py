import math
import random
import matplotlib.pyplot as plt

THRESHOLD = 1e-4
CHANCE = 10
COSZERO = 1.0 - 1e-12


def Rspecular(n1,n2):
    """Compute specular Fresnel reflectance at launch."""
    temp=(n1-n2)/(n1+n2)
    return temp*temp


def RFresnel(n1,n2,ca1):
    """
    General Fresnel reflectance formula.
    ca1 is the cosine of the angle of incidence.
    Returns the reflectance r and the cosine of the transmission angle ca2.
    """
    if n1==n2:
        return 0.0,ca1

    if ca1>0.99999:
        temp=(n1-n2)/(n1+n2)
        return temp*temp,ca1

    sa1=math.sqrt(1.0-ca1*ca1)
    sa2=n1*sa1/n2

    if sa2>=1.0:
        return 1.0,0.0

    ca2=math.sqrt(1.0-sa2*sa2)
    cap=ca1*ca2-sa1*sa2
    cam=ca1*ca2+sa1*sa2
    sap=sa1*ca2+ca1*sa2
    sam=sa1*ca2-ca1*sa2

    r = 0.5*sam*sam*(cam*cam+cap*cap)/(sap*sap*cam*cam)
    return r, ca2


def LaunchPhoton(n_ambient, n_tissue):
    """Initialize a photon packet."""
    rsp = Rspecular(n_ambient, n_tissue)
    photon = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "ux": 0.0,
        "uy": 0.0,
        "uz": 1.0,
        "w": 1.0 - rsp,
        "dead": False,
        "s": 0.0,
        "sleft": 0.0,
    }
    return photon


def StepSizeInTissue():
    """Sample optical depth for the next tissue step."""
    return -math.log(max(random.random(), 1e-100))


def Hop(photon, s):
    """Move the photon by physical distance s."""
    photon["x"] += s*photon["ux"]
    photon["y"] += s*photon["uy"]
    photon["z"] += s*photon["uz"]


def SpinTheta(g):
    """Sample the cosine of the scattering polar angle."""
    if g==0.0:
        return 2.0*random.random()-1.0

    temp = (1.0-g*g)/(1.0-g+2.0*g*random.random())
    cos_theta = (1.0+g*g-temp*temp)/(2.0*g)
    return max(-1.0,min(1.0,cos_theta))


def Spin(photon, g):
    """Rotate the photon direction using Henyey-Greenstein scattering."""
    cos_theta=SpinTheta(g)
    sin_theta=math.sqrt(max(0.0,1.0-cos_theta*cos_theta))
    psi=2.0*math.pi*random.random()
    cos_psi=math.cos(psi)
    sin_psi=math.sin(psi)

    ux=photon["ux"]
    uy=photon["uy"]
    uz=photon["uz"]

    if abs(uz)>COSZERO:
        ux_new=sin_theta*cos_psi
        uy_new=sin_theta*sin_psi
        uz_new=cos_theta*(1.0 if uz>0.0 else -1.0)
    else:
        temp=math.sqrt(1.0-uz*uz)
        ux_new=(sin_theta*(ux*uz*cos_psi-uy*sin_psi))/temp+ux*cos_theta
        uy_new=(sin_theta*(uy*uz*cos_psi+ux*sin_psi))/temp+uy*cos_theta
        uz_new=-sin_theta*cos_psi*temp+uz*cos_theta

    photon["ux"]=ux_new
    photon["uy"]=uy_new
    photon["uz"]=uz_new


def Drop(photon,mu_a,mu_t,results):
    """
    Remove absorbed weight and score it into the depth-resolved fluence bins.
    Note that fluence here actually refers to absorption and needs to be divided by mu_a later.
    """
    dw=photon["w"]*mu_a/mu_t
    photon["w"] -= dw

    iz=int(photon["z"]/results["dz"])
    if 0 <= iz < len(results["fluence"]):
        results["fluence"][iz] += dw


def Roulette(photon):
    """Terminate very low-weight photons."""
    if photon["w"]>=THRESHOLD:
        return

    if random.random()<1.0/CHANCE:
        photon["w"]*=CHANCE
    else:
        photon["w"]=0.0
        photon["dead"]=True


def CrossUpOrNot(photon,n_tissue,n_ambient,results):
    """Handle boundary crossing at the top surface (z=0)."""
    ca1=abs(photon["uz"])
    r,ca2=RFresnel(n_tissue,n_ambient,ca1)

    if random.random()<=r:
        photon["uz"]=-photon["uz"]
    else:
        photon["dead"] = True
        results["total_reflectance"] += photon["w"]
        photon["w"] = 0.0


def main_photon_loop(photon,n_tissue,n_ambient,mu_a,mu_s,g,results):
    """
    Track one photon until it dies.

    StepSizeInTissue returns optical depth; this loop converts it to
    physical distance using mu_t.
    """
    mu_t=mu_a+mu_s

    while not photon["dead"]:
        if photon["sleft"]==0.0:
            photon["s"]=StepSizeInTissue()
        else:
            photon["s"]=photon["sleft"]
            photon["sleft"]=0.0

        step_physical=photon["s"]/mu_t
        hit_boundary=False
        d_boundary=0.0

        if photon["uz"]<0.0:
            d_boundary=photon["z"]/abs(photon["uz"])
            if step_physical>d_boundary:
                hit_boundary=True

        if hit_boundary:
            Hop(photon,d_boundary)
            photon["sleft"] = (step_physical - d_boundary) * mu_t
            CrossUpOrNot(photon, n_tissue, n_ambient, results)
        else:
            Hop(photon,step_physical)
            Drop(photon,mu_a,mu_t,results)

            if not photon["dead"]:
                Spin(photon,g)
                Roulette(photon)


def run_simulation(n_rel,mu_a,mu_s,g,N_photons,dz=0.01,nz=500):
    """Run the Monte Carlo simulation and return reflectance and fluence (absorption)."""
    n_ambient = 1.0
    n_tissue = n_rel * n_ambient

    results = {
        "total_reflectance": 0.0,
        "fluence": [0.0] * nz,
        "dz": dz,
    }

    for _ in range(N_photons):
        photon=LaunchPhoton(n_ambient,n_tissue)
        main_photon_loop(photon,n_tissue,n_ambient,mu_a,mu_s,g,results)

    norm=dz*N_photons
    results["fluence"]=[value/norm for value in results["fluence"]]
    return results


def run_batch(n_rel,mu_a,mu_s,g,photons_per_run,n_runs,dz=0.01,nz=500):
    """Run multiple independent simulations and summarize reflectance statistics."""
    diffuse_reflectances=[]
    fluence_runs=[]

    for _ in range(n_runs):
        results=run_simulation(n_rel,mu_a,mu_s,g,photons_per_run,dz=dz,nz=nz)
        diffuse_reflectances.append(results["total_reflectance"]/photons_per_run)
        fluence_runs.append(results["fluence"])

    mean_diffuse=sum(diffuse_reflectances)/n_runs
    if n_runs>1:
        variance=sum((value-mean_diffuse)**2 for value in diffuse_reflectances)/(n_runs-1)
        stderr=math.sqrt(variance/n_runs)
    else:
        stderr=0.0

    mean_fluence = [
        sum(fluence[i] for fluence in fluence_runs) / n_runs for i in range(nz)
    ]
    specular = Rspecular(1.0, n_rel)

    return {
        "diffuse_reflectances": diffuse_reflectances,
        "mean_diffuse_reflectance": mean_diffuse,
        "specular_reflectance": specular,
        "mean_total_reflectance": mean_diffuse + specular,
        "reflectance_stderr": stderr,
        "mean_fluence": mean_fluence,
        "dz": dz,
        "photons_per_run": photons_per_run,
        "n_runs": n_runs,
    }

mu_a = 10.0
summary = run_batch(1.5, mu_a, 90.0, 0.0, photons_per_run=5000, n_runs=10)
print(f"Diffuse reflectance per run: {summary['diffuse_reflectances']}")
print(f"Mean diffuse reflectance: {summary['mean_diffuse_reflectance']:.6f}")
print(f"Specular reflectance: {summary['specular_reflectance']:.6f}")
print(f"Mean total reflectance: {summary['mean_total_reflectance']:.6f}")
print(f"Standard error: {summary['reflectance_stderr']:.6f}")

depth_resolved_fluence = [f / mu_a for f in summary['mean_fluence']]
print(f"\nDepth-resolved Fluence:\n{depth_resolved_fluence}")