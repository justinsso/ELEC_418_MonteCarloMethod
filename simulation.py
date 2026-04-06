import math
import random

def Rspecular(n1, n2):
    """Compute specular Fresnel reflectance at launch."""
    temp = (n1 - n2) / (n1 + n2)
    return temp * temp

def RFresnel(n1, n2, ca1):
    """
    General Fresnel reflectance formula.
    ca1 is the cosine of the angle of incidence.
    Returns the reflectance r and the cosine of the transmission angle ca2.
    """
    if n1 == n2:
        return 0.0, ca1

    if ca1 > 0.99999: # Normal incidence
        temp = (n1 - n2) / (n1 + n2)
        return temp * temp, ca1

    sa1 = math.sqrt(1.0 - ca1 * ca1)
    sa2 = n1 * sa1 / n2

    if sa2 >= 1.0: # Total internal reflection
        return 1.0, 0.0

    ca2 = math.sqrt(1.0 - sa2 * sa2)
    cap = ca1 * ca2 - sa1 * sa2 # cos(a1 + a2)
    cam = ca1 * ca2 + sa1 * sa2 # cos(a1 - a2)
    sap = sa1 * ca2 + ca1 * sa2 # sin(a1 + a2)
    sam = sa1 * ca2 - ca1 * sa2 # sin(a1 - a2)

    r = 0.5 * sam * sam * (cam * cam + cap * cap) / (sap * sap * cam * cam)
    return r, ca2

def LaunchPhoton(n_ambient, n_tissue):
    """Initialize a photon packet."""
    rsp = Rspecular(n_ambient, n_tissue)
    photon = {
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'ux': 0.0,
        'uy': 0.0,
        'uz': 1.0,
        'w': 1.0 - rsp,
        'dead': False,
        's': 0.0,
        'sleft': 0.0
    }
    return photon

def CrossUpOrNot(photon, n_tissue, n_ambient, results):
    """Handle boundary crossing at the top surface (z=0)."""
    ca1 = abs(photon['uz'])
    r, ca2 = RFresnel(n_tissue, n_ambient, ca1)

    if random.random() <= r:
        # Internally reflected
        photon['uz'] = -photon['uz']
    else:
        # Transmitted (escapes the tissue)
        photon['dead'] = True
        results['total_reflectance'] += photon['w']
        photon['w'] = 0.0

def main_photon_loop(photon, n_tissue, n_ambient, mu_a, mu_s, g, results):
    """
    The main loop tracking the photon until it is dead.
    Assumes Person A has defined: StepSizeInTissue, Hop, Drop, Spin, and Roulette.
    """
    mu_t = mu_a + mu_s

    while not photon['dead']:
        # 1. Generate step size
        if photon['sleft'] == 0.0:
            photon['s'] = StepSizeInTissue() # Called from Person A
        else:
            photon['s'] = photon['sleft']
            photon['sleft'] = 0.0

        step_physical = photon['s'] / mu_t
        hit_boundary = False
        d_boundary = 0.0

        # 2. Check for boundary hit
        # In a semi infinite medium, the only boundary is z = 0.
        # A photon can only hit this boundary if it is moving upwards (uz < 0).
        if photon['uz'] < 0.0:
            d_boundary = photon['z'] / abs(photon['uz'])
            if step_physical > d_boundary:
                hit_boundary = True

        # 3. Move and process interactions
        if hit_boundary:
            Hop(photon, d_boundary) # Called from Person A
            photon['sleft'] = (step_physical - d_boundary) * mu_t
            CrossUpOrNot(photon, n_tissue, n_ambient, results)
        else:
            Hop(photon, step_physical) # Called from Person A
            Drop(photon, mu_a, mu_t, results) # Called from Person A

            if not photon['dead']:
                Spin(photon, g) # Called from Person A
                Roulette(photon) # Called from Person A