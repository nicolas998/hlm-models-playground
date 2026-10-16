"""Hillslope Link Model (HLM) right-hand-side definitions.

Every model is registered in :data:`MODELS` by name, so the notebook can pick
and test any of them interchangeably::

    from models import MODELS, get_model
    model = get_model("254_toplayer")
    sol = solve_ivp(model.dydt, ..., args=(params, global_params, forcing))

Adding a new model
------------------
Each model is built from a *flux function* and a *RHS function*:

* ``fluxes(t, y, params, global_params, forcing) -> dict``
      returns every internal flux/rate (q_pl, q_pt, q_ts, q_sl, evaporation,
      k_t, ...). This is the single source of truth.
* ``dydt(t, y, params, global_params, forcing, parents_q=0, parents_qb=0)``
      assembles dy/dt from those fluxes; this is what ``solve_ivp`` integrates.

Register the pair with :func:`register_model`. Because the RHS and the flux
extractor share the same flux function, the post-run flux analysis is always
consistent with the integrated solution. Use :func:`flux_timeseries` to evaluate
the fluxes over a whole solution.

RHS contract
------------
* ``t``              : time [min] (used to sample the forcing)
* ``y``              : state vector, length ``dim``
* ``params``         : per-link params from ``setup.precalculations``
                       -> [L, A_h, invtau, k_2, k_i, c_1, c_2]
* ``global_params``  : global parameter vector (see setup docstring)
* ``forcing``        : callable ``forcing(t) -> (rain_mm_hr, epot_mm_month)``
                       or a 2-tuple of constants
* ``parents_q``      : summed upstream channel discharge [m^3/s]
* ``parents_qb``     : summed upstream base flow         [m^3/s]

Flux dict keys (254 family), all hillslope rates in [m/min]
-----------------------------------------------------------
    rain_in : rainfall reaching ponded storage  (rain_mm_hr * c_1)
    e_p, e_t, e_s : evaporation from ponded / top / sub-surface
    k_t     : effective infiltration coefficient        [1/min]
    q_pl    : ponded   -> channel
    q_pt    : ponded   -> top layer
    q_ts    : top      -> sub-surface
    q_sl    : sub-sfc  -> channel
"""

from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np


# ----------------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class HLMModel:
    """Metadata + right-hand side (+ optional flux extractor) for one variant."""
    name: str
    dydt: Callable
    dim: int
    state_names: Tuple[str, ...]
    description: str = ""
    fluxes: Optional[Callable] = None     # fluxes(t, y, params, gp, forcing) -> dict


MODELS = {}


def register_model(name, dim, state_names, description="", fluxes=None):
    """Decorator: register a dydt function as a named model.

    ``fluxes`` is an optional companion function returning a dict of internal
    fluxes for the same state; if given, :func:`flux_timeseries` can reconstruct
    the fluxes over a solution.
    """
    def deco(fn):
        MODELS[name] = HLMModel(
            name=name, dydt=fn, dim=dim,
            state_names=tuple(state_names), description=description,
            fluxes=fluxes,
        )
        return fn
    return deco


def get_model(name):
    """Look up a registered model by name (raises if unknown)."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {list(MODELS)}")
    return MODELS[name]


def flux_timeseries(model, t, y, params, global_params, forcing):
    """Evaluate a model's fluxes at every time point of a solution.

    Parameters
    ----------
    model : HLMModel
        A model with a non-None ``fluxes`` companion.
    t : array_like, shape (N,)
        Times [min] (e.g. ``sol.t``).
    y : array_like, shape (dim, N)
        States, one column per time (e.g. ``sol.y``).
    params, global_params, forcing :
        Same objects passed to the integration.

    Returns
    -------
    dict[str, numpy.ndarray]
        One array of shape (N,) per flux key. The set of keys is whatever the
        model's ``fluxes`` function returns.
    """
    if model.fluxes is None:
        raise ValueError(f"model {model.name!r} has no flux function registered")
    t = np.atleast_1d(np.asarray(t, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if y.shape[0] != model.dim and y.shape[1] == model.dim:
        y = y.T                                   # accept (N, dim) too
    n = t.size
    out = {}
    for k in range(n):
        f = model.fluxes(t[k], y[:, k], params, global_params, forcing)
        for key, val in f.items():
            if key not in out:
                out[key] = np.empty(n)
            out[key][k] = val
    return out


def _read_forcing(forcing, t):
    """Sample forcing at time t and convert e_pot to [m/min]."""
    rain_mm_hr, epot_mm_month = forcing(t) if callable(forcing) else forcing
    e_pot = epot_mm_month * (1e-3 / (30.0 * 24.0 * 60.0))   # mm/month -> m/min
    return rain_mm_hr, e_pot


def _evaporation(s_p, s_t, s_s, e_pot, S_L, h_b):
    """Shared evaporation partitioning used by the 254 family."""
    Corr = s_p + s_t / S_L + s_s / (h_b - S_L)
    if e_pot > 0.0 and Corr > 1e-12:
        e_p = s_p * e_pot / Corr
        e_t = s_t / S_L * e_pot / Corr
        e_s = s_s / (h_b - S_L) * e_pot / Corr
        return e_p, e_t, e_s
    return 0.0, 0.0, 0.0


def _assemble_254_rhs(y, params, global_params, f, parents_q, parents_qb):
    """Assemble the (shared) 254-family dy/dt from a flux dict ``f``."""
    lambda_1 = global_params[1]
    v_B      = global_params[11]
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params

    q   = y[0]
    q_b = max(0.001, y[6])

    ans = np.zeros(7)
    # channel discharge
    ans[0] = -q + (f["q_pl"] + f["q_sl"]) * c_2 + parents_q
    ans[0] = invtau * (q ** lambda_1) * ans[0]
    # hillslope storages
    ans[1] = f["rain_in"] - f["q_pl"] - f["q_pt"] - f["e_p"]
    ans[2] = f["q_pt"] - f["q_ts"] - f["e_t"]
    ans[3] = f["q_ts"] - f["q_sl"] - f["e_s"]
    # diagnostics
    ans[4] = f["rain_in"]          # accumulated precip
    ans[5] = f["q_pl"]             # accumulated runoff to channel
    # base flow
    ans[6] = f["q_sl"] * A_h - q_b * 60.0 + parents_qb * 60.0
    ans[6] *= v_B / L
    return ans


# ----------------------------------------------------------------------------
# Model 254 -- "Top Layer" (the supplied C kernel)
# ----------------------------------------------------------------------------
def fluxes_254(t, y, params, global_params, forcing):
    k_3      = global_params[4]      # [1/min]
    h_b      = global_params[6]      # [m]
    S_L      = global_params[7]      # [m]
    A        = global_params[8]
    B        = global_params[9]
    exponent = global_params[10]

    rain_mm_hr, e_pot = _read_forcing(forcing, t)
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params
    s_p, s_t, s_s = y[1], y[2], y[3]

    e_p, e_t, e_s = _evaporation(s_p, s_t, s_s, e_pot, S_L, h_b)

    # saturation-excess infiltration into top layer
    pow_term = (1.0 - s_t / S_L) ** exponent if (1.0 - s_t / S_L) > 0.0 else 0.0
    k_t = (A + B * pow_term) * k_2

    return {
        "rain_in": rain_mm_hr * c_1,
        "e_p": e_p, "e_t": e_t, "e_s": e_s,
        "k_t": k_t,
        "q_pl": k_2 * s_p,          # ponded  -> channel
        "q_pt": k_t * s_p,          # ponded  -> top
        "q_ts": k_i * s_t,          # top     -> sub-surface
        "q_sl": k_3 * s_s,          # sub-sfc -> channel
    }


@register_model(
    "254_toplayer", dim=7,
    state_names=("q", "s_p", "s_t", "s_s", "s_precip", "V_r", "q_b"),
    description="Top Layer model (Asynch 254): ponded / top / sub-surface "
                "storages with saturation-excess infiltration.",
    fluxes=fluxes_254,
)
def model_254(t, y, params, global_params, forcing,
              parents_q=0.0, parents_qb=0.0):
    f = fluxes_254(t, y, params, global_params, forcing)
    return _assemble_254_rhs(y, params, global_params, f, parents_q, parents_qb)


# ----------------------------------------------------------------------------
# Example variant -- constant infiltration (no saturation-excess non-linearity)
# ----------------------------------------------------------------------------
def fluxes_254_const_infil(t, y, params, global_params, forcing):
    k_3 = global_params[4]
    h_b = global_params[6]
    S_L = global_params[7]
    A   = global_params[8]

    rain_mm_hr, e_pot = _read_forcing(forcing, t)
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params
    s_p, s_t, s_s = y[1], y[2], y[3]

    e_p, e_t, e_s = _evaporation(s_p, s_t, s_s, e_pot, S_L, h_b)
    k_t = A * k_2                  # <-- the only change vs. model 254

    return {
        "rain_in": rain_mm_hr * c_1,
        "e_p": e_p, "e_t": e_t, "e_s": e_s,
        "k_t": k_t,
        "q_pl": k_2 * s_p,
        "q_pt": k_t * s_p,
        "q_ts": k_i * s_t,
        "q_sl": k_3 * s_s,
    }


@register_model(
    "254_const_infil", dim=7,
    state_names=("q", "s_p", "s_t", "s_s", "s_precip", "V_r", "q_b"),
    description="Like 254 but k_t = A*k_2 (constant infiltration, no "
                "saturation-excess term).",
    fluxes=fluxes_254_const_infil,
)
def model_254_const_infil(t, y, params, global_params, forcing,
                          parents_q=0.0, parents_qb=0.0):
    f = fluxes_254_const_infil(t, y, params, global_params, forcing)
    return _assemble_254_rhs(y, params, global_params, f, parents_q, parents_qb)


# ----------------------------------------------------------------------------
# Model 255 -- Top Layer with NON-LINEAR sub-surface -> channel release
# ----------------------------------------------------------------------------
def fluxes_255(t, y, params, global_params, forcing):
    k_3         = global_params[4]      # [1/min]
    h_b         = global_params[6]      # [m]
    S_L         = global_params[7]      # [m]
    A           = global_params[8]
    B           = global_params[9]
    exponent    = global_params[10]
    SsMax       = global_params[12]     # [m]
    SsMin       = global_params[13]     # [m]
    Ss_Coeff    = global_params[14]     # [1/m]
    Ss_exponent = global_params[15]     # [-]

    rain_mm_hr, e_pot = _read_forcing(forcing, t)
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params
    s_p, s_t, s_s = y[1], y[2], y[3]

    e_p, e_t, e_s = _evaporation(s_p, s_t, s_s, e_pot, S_L, h_b)

    pow_term_pt = (1.0 - s_t / S_L) ** exponent if (1.0 - s_t / S_L) > 0.0 else 0.0
    k_t = (A + B * pow_term_pt) * k_2

    pow_term_sl = (1.0 - (s_s-SsMax) / h_b) ** Ss_exponent if (1.0 - (s_s-SsMax) / h_b) > 0.0 else 0.0
    k_s = pow_term_sl * k_2 * Ss_Coeff

    # non-linear baseflow from sub-surface storage
    q_sl = 0.0
    if s_s > SsMin:
        q_sl = k_3 * s_s#(s_s - SsMin)
    if s_s > SsMax:
        q_sl += Ss_Coeff * (s_s - SsMax) * np.exp(Ss_exponent * (s_s - SsMax)) # k_s * (s_s-SsMax) #

    return {
        "rain_in": rain_mm_hr * c_1,
        "e_p": e_p, "e_t": e_t, "e_s": e_s,
        "k_t": k_t,
        "q_pl": k_2 * s_p,
        "q_pt": k_t * s_p,
        "q_ts": k_i * s_t,
        "q_sl": q_sl,
    }


@register_model(
    "255_NonLinearSub", dim=7,
    state_names=("q", "s_p", "s_t", "s_s", "s_precip", "V_r", "q_b"),
    description="Top Layer model (Asynch 255): like 254 but with a non-linear "
                "sub-surface -> channel release.",
    fluxes=fluxes_255,
)
def model_255(t, y, params, global_params, forcing,
              parents_q=0.0, parents_qb=0.0):
    f = fluxes_255(t, y, params, global_params, forcing)
    return _assemble_254_rhs(y, params, global_params, f, parents_q, parents_qb)


# ----------------------------------------------------------------------------
# Collie River Basin 3 (MARRMoT-style) -- soil / ponded / groundwater storages
# ----------------------------------------------------------------------------
# Smoothing of the saturation-excess threshold (MARRMoT smoothThreshStorage).
CRB3_SMOOTH_R = 0.01
CRB3_SMOOTH_E = 5.0


def _smooth_excess(S, Smax, r=CRB3_SMOOTH_R, e=CRB3_SMOOTH_E):
    """Fraction (0..1) of input that becomes excess as S approaches Smax."""
    x = (S - Smax + r * e * Smax) / (r * Smax)
    x = min(max(x, -50.0), 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def fluxes_crb3(t, y, params, global_params, forcing):
    A      = global_params[0]
    B      = global_params[1]
    lam    = global_params[2]
    Ssmax  = global_params[3]       # [m]
    Sfc    = global_params[4] * Ssmax   # fraction -> [m]
    M      = global_params[5]
    A2     = global_params[6]
    B2     = global_params[7]

    rain_mm_hr, e_pot = _read_forcing(forcing, t)
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params
    s_s, s_p, s_g = max(y[1], 0.0), max(y[2], 0.0), max(y[3], 0.0)

    P = rain_mm_hr * c_1

    # evaporation: bare soil and vegetation
    e_b = s_s / Ssmax * (1.0 - M) * e_pot
    e_v = M * e_pot if s_s > Sfc else s_s / Sfc * M * e_pot

    # saturation excess -> ponded storage
    q_sp = P * _smooth_excess(s_s, Ssmax)

    # drainage above field capacity, split to groundwater / channel
    q_ss = A * (s_s - Sfc) ** B if s_s > Sfc else 0.0
    q_sg = lam * q_ss
    q_sL = (1.0 - lam) * q_ss

    q_pL = k_2 * s_p
    q_gL = A2 * s_g ** B2

    return {
        "rain_in": P,
        "e_b": e_b, "e_v": e_v,
        "q_sp": q_sp, "q_ss": q_ss, "q_sg": q_sg, "q_sL": q_sL,
        "q_pL": q_pL, "q_gL": q_gL,
        "q_out": (q_pL + q_gL + q_sL) * c_2,    # hillslope outflow [m^3/s]
    }


@register_model(
    "CRB3", dim=4,
    state_names=("Q", "s_s", "s_p", "s_g"),
    description="Collie River Basin 3: soil (Ss), ponded (Sp) and groundwater "
                "(Sg) storages. Q is cumulative runoff [m]; streamflow is the "
                "'q_out' flux [m^3/s].",
    fluxes=fluxes_crb3,
)
def model_crb3(t, y, params, global_params, forcing,
               parents_q=0.0, parents_qb=0.0):
    # parents_q / parents_qb are ignored: Q is not routed in this model.
    f = fluxes_crb3(t, y, params, global_params, forcing)
    ans = np.zeros(4)
    ans[0] = f["q_pL"] + f["q_gL"] + f["q_sL"]                          # Q
    ans[1] = f["rain_in"] - f["e_v"] - f["e_b"] - f["q_sp"] - f["q_ss"]  # Ss
    ans[2] = f["q_sp"] - f["q_pL"]                                      # Sp
    ans[3] = f["q_sg"] - f["q_gL"]                                      # Sg
    return ans
