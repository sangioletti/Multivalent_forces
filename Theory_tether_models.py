# Theory_tether_models.py -- ADDITION to Theory_v9_mp.py (the latter is not modified).
#
# Alternative tether (ligand) models for the configurational part of the bond
# free energy, DG_cnf(d), and the associated single-bond restoring force, to be
# used in place of the Gaussian (harmonic) spring of the main text.  Here d is
# the ligand grafting-point -> receptor distance, b = a_K the Kuhn length,
# L = n_K a_K the contour length and R_ee^2 = n_K a_K^2 = L b.
#
#   'gaussian' : DG = 1/2 k_eff d^2,  k_eff = 3 kBT / R_ee^2             (default; unchanged)
#
#   'fjc'      : freely-jointed chain, inverse-Langevin force law with the
#                Pade approximant of Cohen (1991), L^{-1}(x) ~ x (3 - x^2)/(1 - x^2):
#                   f(d)  = (kBT/b) L^{-1}(x),            x = d/L
#                   DG(d) = (kBT L/b) [ x^2/2 - ln(1 - x^2) ]
#
#   'wlc'      : worm-like chain, Marko-Siggia (1995) interpolation, l_p = b/2:
#                   f(d)  = (kBT/l_p) [ 1/(4(1-x)^2) - 1/4 + x ]
#                   DG(d) = (kBT L/l_p) [ 1/(4(1-x)) - 1/4 - x/4 + x^2/2 ]
#
# All three models share the same small-extension limit DG -> 3 kBT d^2 / (2 R_ee^2),
# i.e. they coincide with the Gaussian chain for d << L, while the two finitely
# extensible models diverge for d -> L (chi -> 0 and no bond for d >= L).
# The steric (surface-repulsion) term A_steric of Theory_v9_mp is left unchanged.
#
# Usage
# -----
#     import Theory_v9_mp as theory
#     import Theory_tether_models as tm
#     tm.install(theory)                 # patch chi_np, log_chi_np, pureForce_np, _pLEq_mpmath
#     data['tether_model'] = 'wlc'       # 'gaussian' (default) | 'fjc' | 'wlc'
#
# When data['tether_model'] is absent or 'gaussian' the ORIGINAL functions are
# called, so existing scripts are unaffected by the patch.

import numpy as np
import mpmath

_orig = {}

MODELS = ('gaussian', 'fjc', 'wlc')


def _geom(data):
    b = float(data['akuhn'])
    L = float(data['Nkuhn']) * b
    return b, L


def DGcnf(dist, data):
    """Configurational bond free energy at grafting-point distance `dist`."""
    model = data.get('tether_model', 'gaussian')
    kbT = data['kbT']
    dist = np.asarray(dist, dtype=float)
    if model == 'gaussian':
        return 0.5 * data['kEff'] * (dist - data['x0']) ** 2
    b, L = _geom(data)
    x = np.clip(dist / L, 0.0, 1.0 - 1e-12)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if model == 'fjc':
            val = (kbT * L / b) * (0.5 * x**2 - np.log1p(-x**2))
        elif model == 'wlc':
            lp = 0.5 * b
            val = (kbT * L / lp) * (0.25 / (1.0 - x) - 0.25 - 0.25 * x + 0.5 * x**2)
        else:
            raise ValueError(f"Unknown tether_model '{model}', choose from {MODELS}")
    return np.where(dist >= L, np.inf, val)


def tetherForce(dist, data):
    """Restoring force of the tether, same sign convention as Theory_v9_mp.pureForce_np
    (negative = pulls the receptor towards the grafting point).  Zero for dist >= L,
    where no bond can exist (chi = 0)."""
    model = data.get('tether_model', 'gaussian')
    kbT = data['kbT']
    dist = np.asarray(dist, dtype=float)
    if model == 'gaussian':
        return -data['kEff'] * np.abs(dist - data['x0'])
    b, L = _geom(data)
    x = np.clip(dist / L, 0.0, 1.0 - 1e-12)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if model == 'fjc':
            f = (kbT / b) * x * (3.0 - x**2) / (1.0 - x**2)
        elif model == 'wlc':
            lp = 0.5 * b
            f = (kbT / lp) * (0.25 / (1.0 - x)**2 - 0.25 + x)
        else:
            raise ValueError(f"Unknown tether_model '{model}', choose from {MODELS}")
    return -np.where(dist >= L, 0.0, f)


def DGcnf_mp(dist, data):
    """mpmath version of DGcnf (scalar), used only in the high-precision fallback."""
    model = data.get('tether_model', 'gaussian')
    kbT = mpmath.mpf(str(float(data['kbT'])))
    if model == 'gaussian':
        return mpmath.mpf(str(float(data['kEff']))) / 2 * (dist - mpmath.mpf(str(float(data['x0']))))**2
    b, L = _geom(data)
    b = mpmath.mpf(str(b)); L = mpmath.mpf(str(L))
    if dist >= L:
        return mpmath.inf
    x = dist / L
    if model == 'fjc':
        return kbT * L / b * (x**2 / 2 - mpmath.log(1 - x**2))
    lp = b / 2
    return kbT * L / lp * (mpmath.mpf(1) / (4 * (1 - x)) - mpmath.mpf(1) / 4 - x / 4 + x**2 / 2)


# ----------------------------------------------------------------------
#  Patched versions of the Theory_v9_mp primitives
# ----------------------------------------------------------------------

def _chi_np(r, z, data):
    if data.get('tether_model', 'gaussian') == 'gaussian':
        return _orig['chi_np'](r, z, data)
    dist = np.sqrt(np.asarray(r, dtype=float)**2 + np.asarray(z, dtype=float)**2)
    arg = -(data['DG0'] + DGcnf(dist, data)) / data['kbT']
    if np.any(arg > 700):
        raise ValueError(f"Unsolvable Overflow in chi_np: arg = {arg}")
    with np.errstate(under='ignore'):
        return np.exp(np.where(np.isneginf(arg), -np.inf, arg))


def _log_chi_np(r, z, data):
    if data.get('tether_model', 'gaussian') == 'gaussian':
        return _orig['log_chi_np'](r, z, data)
    dist = np.sqrt(np.asarray(r, dtype=float)**2 + np.asarray(z, dtype=float)**2)
    return -(data['DG0'] + DGcnf(dist, data)) / data['kbT']


def _pureForce_np(r, z, data):
    if data.get('tether_model', 'gaussian') == 'gaussian':
        return _orig['pureForce_np'](r, z, data)
    dist = np.sqrt(np.asarray(r, dtype=float)**2 + np.asarray(z, dtype=float)**2)
    return tetherForce(dist, data)


def _pLEq_mpmath(z, data):
    if data.get('tether_model', 'gaussian') == 'gaussian':
        return _orig['_pLEq_mpmath'](z, data)
    T = _orig['module']
    NL = T._mp(data['NL']); sigma = T._mp(data['sigma']); rMax = T._mp(data['rMax'])
    z_mp = T._mp(z); DG0 = T._mp(data['DG0']); kbT = T._mp(data['kbT'])

    def chi_m(r):
        dist = mpmath.sqrt(r**2 + z_mp**2)
        dg = DGcnf_mp(dist, data)
        if dg == mpmath.inf:
            return mpmath.mpf(0)
        return mpmath.exp(-(DG0 + dg) / kbT)

    def f(pL):
        def integrand(r):
            x = chi_m(r) * pL
            return 2 * mpmath.pi * r * sigma * x / (1 + NL * x)
        integral = mpmath.quad(integrand, [mpmath.mpf('1e-12'), rMax])
        return pL + mpmath.re(integral) - 1

    a = mpmath.mpf('1e-100'); b = mpmath.mpf('1')
    for _ in range(300):
        mid = (a + b) / 2
        fm = f(mid)
        if mpmath.fabs(fm) < mpmath.mpf('1e-40') or (b - a) / 2 < mpmath.mpf('1e-45'):
            return T._float(mid)
        if f(a) * fm < 0:
            b = mid
        else:
            a = mid
    return T._float((a + b) / 2)


def install(theory_module):
    """Patch the given Theory_v9_mp module in place (idempotent)."""
    if _orig.get('module') is theory_module:
        return
    _orig['module'] = theory_module
    for name in ('chi_np', 'log_chi_np', 'pureForce_np', '_pLEq_mpmath'):
        _orig[name] = getattr(theory_module, name)
    theory_module.chi_np = _chi_np
    theory_module.log_chi_np = _log_chi_np
    theory_module.pureForce_np = _pureForce_np
    theory_module._pLEq_mpmath = _pLEq_mpmath


def uninstall():
    T = _orig.get('module')
    if T is None:
        return
    for name in ('chi_np', 'log_chi_np', 'pureForce_np', '_pLEq_mpmath'):
        setattr(T, name, _orig[name])
    _orig.clear()
