# Theory_independent.py -- ADDITION to Theory_v9_mp.py (the latter is not modified).
#
# Independent-binders limit (reply to Reviewer 1, comment 2).
#
# In the main text every ligand is grafted (effectively) at the nanoparticle
# centre, so that all N_L ligands compete for the same N_R receptors.  The
# opposite, analytically tractable limit is the one in which the grafting
# distance is much larger than the tether length, so that each ligand sees its
# own, non-overlapping patch of receptors.  In this limit:
#
#   * the self-consistent equations for p_L, p_R are those of Theory_v9_mp with
#     N_L = 1 (one ligand facing N_R receptors in its patch):
#         p_L(z) + int 2 pi r sigma chi p_L / (1 + chi p_L) dr = 1
#         p_R(r,z) = 1 / (1 + p_L chi)
#   * the force on a receptor at (r,z) is   beta F_alpha = p_R p_L grad_alpha chi
#     and is identical in every patch;
#   * ligands are statistically independent, so bond free energies add:
#         A_bond^indep(z) = -kBT ln{ exp[-N_L beta A_1(z)] - 1 },
#     where A_1(z) is the (single-ligand) bond free energy before removal of
#     the fully unbound state; A_steric is unchanged (already proportional to N_L);
#   * the receptor-population average F_alpha = f/N_R int 2 pi r sigma F_b dr is
#     the same whether taken over one patch (N_R receptors) or over all N_L
#     patches (N_L N_R receptors).
#
# Public API (same signature / return convention as Theory_v9_mp):
#     averageForce_indep(direction, data)  ->  (result, result2)
# with result2 the population average including the occupied-site fraction f,
# exactly as used by the Run_Fig*.py scripts.

import numpy as np
from scipy.integrate import simpson as sp_simpson
import Theory_v9_mp as T

_SKIP = ('_pL_cache', 'boundPartition', 'PBoundZ', '_data1',
         'boundPartition_indep', 'PBoundZ_indep')


def _single(data):
    """Return (and cache) the N_L = 1 copy of `data` used for one patch."""
    d1 = data.get('_data1')
    if d1 is None:
        d1 = {k: v for k, v in data.items() if k not in _SKIP}
        d1['NL'] = 1
        d1['_pL_cache'] = {}
        data['_data1'] = d1
    return d1


def ABonds_indep(z, data):
    kbT = data['kbT']; NL = data['NL']
    d1 = _single(data)
    pL = T.pLEq(z, d1)
    if pL == 0:
        return -np.inf
    part1 = kbT * (np.log(pL) + 0.5 * (1 - pL))          # N_L = 1
    part2 = T.integralPR(z, pL, d1)
    exponent = -NL * (part1 + part2) / kbT                 # ligands independent -> energies add
    if exponent < 0:
        raise ValueError(f"Numerical error, Abond >= 0 expected but exponent = {exponent} < 0")
    if exponent > 700:
        return -np.inf
    qBound = np.exp(exponent) - 1.0
    return -kbT * np.log(qBound) if qBound > 0 else np.inf


def ATot_indep(z, data):
    return ABonds_indep(z, data) + T.ARep(z, data)


def Omega_indep(data):
    if 'boundPartition_indep' in data:
        return data['boundPartition_indep']
    zSamples = data['zSamples']; kbT = data['kbT']
    myA = np.zeros(len(zSamples))
    for i, zi in enumerate(zSamples):
        a_val = ATot_indep(zi, data)
        myA[i] = 0.0 if np.isposinf(a_val) else np.exp(np.clip(-a_val / kbT, -700, 700))
    result = sp_simpson(myA, x=zSamples)
    data['boundPartition_indep'] = result
    return result


def pCondBound_indep(z, data):
    den = Omega_indep(data)
    if den <= 0:
        return 0.0
    a_val = ATot_indep(z, data)
    if np.isposinf(a_val):
        return 0.0
    return np.exp(np.clip(-a_val / data['kbT'], -700, 700)) / den


def forceSingle_indep(r, z, direction, data):
    """Force on a receptor at (r,z) in a patch: N_L = 1 expression."""
    return T.forceSingle(r, z, direction, _single(data))


def forceBound_indep(r, direction, data):
    zSamples = data['zSamples']
    myForce = forceSingle_indep(r, zSamples, direction, data)
    if 'PBoundZ_indep' not in data:
        data['PBoundZ_indep'] = np.array([pCondBound_indep(zi, data) for zi in zSamples])
    return sp_simpson(myForce * data['PBoundZ_indep'], x=zSamples)


def fractionBound_indep(data):
    """Occupied-site fraction f with the independent-binders partition function."""
    rV = data['rV']; rNP = data['rNP']
    nSites = max(2, int(4.0 * (rV / rNP) ** 2))
    As = 4 * np.pi * rV**2 / nSites
    d = {k: v for k, v in data.items() if k not in _SKIP}
    d['Kbind'] = As * Omega_indep(data)
    return T.fractionBound(d)


def averageForce_indep(direction, data):
    totRec = T.totRecAds(data)
    rSamples = data['rSamples']; sigma = data['sigma']
    myForce = np.array([forceBound_indep(ri, direction, data) for ri in rSamples])
    finalIntegral = sp_simpson(2 * np.pi * rSamples * sigma * myForce, x=rSamples)
    frac = fractionBound_indep(data)
    result = (1.0 / totRec) * finalIntegral if totRec > 0 else 0.0
    result2 = (frac / totRec) * finalIntegral if totRec > 0 else 0.0
    return result, result2
