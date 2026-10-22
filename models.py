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
from scipy.integrate import solve_ivp

from setup import (
    precalculations, precalculations_crb3, precalculations_nzv1,
    precalculations_constant_runoff,
    GLOBAL_PARAMS_254, GLOBAL_PARAMS_255, GLOBAL_PARAMS_CRB3, GLOBAL_PARAMS_NZV1,
    GLOBAL_PARAMS_CONSTANT_RUNOFF,
)


# ----------------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class HLMModel:
    """Metadata + right-hand side (+ optional flux extractor) for one variant.

    The fields after ``fluxes`` make a model runnable by :func:`simulate` (and
    therefore by the web app, see ``app.py``); a model is "runnable" when
    ``precalc``, ``default_params``, ``y0`` and ``streamflow`` are all set.
    """
    name: str
    dydt: Callable
    dim: int
    state_names: Tuple[str, ...]
    description: str = ""
    fluxes: Optional[Callable] = None     # fluxes(t, y, params, gp, forcing) -> dict
    precalc: Optional[Callable] = None    # precalc(L, A_h, A_i, gp) -> params
    param_names: Tuple[Tuple[str, str], ...] = ()   # (name, unit) per gp index
    default_params: Optional[Tuple[float, ...]] = None
    y0: Optional[Tuple[float, ...]] = None          # default initial state
    storages: Tuple[str, ...] = ()        # state names that are storages [m]
    streamflow: Optional[str] = None      # state name or flux key, [m^3/s]
    # Equations, for display (the app's "Equations" view):
    #   balance   : {storage: (inflow flux keys, outflow flux keys)}
    #   flux_eqs  : {flux key: (LaTeX symbol, LaTeX right-hand side)}
    #   extra_eqs : ((LaTeX lhs, LaTeX rhs), ...) definitions / routing
    balance: Optional[dict] = None
    flux_eqs: Optional[dict] = None
    extra_eqs: Tuple[Tuple[str, str], ...] = ()
    #   diagram   : {flux key: (from, to) or None}, overrides of flow_edges()
    diagram: Optional[dict] = None

    @property
    def runnable(self):
        return None not in (self.precalc, self.default_params, self.y0,
                            self.streamflow)


MODELS = {}


def register_model(name, dim, state_names, description="", fluxes=None,
                   precalc=None, param_names=(), default_params=None, y0=None,
                   storages=(), streamflow=None, balance=None,
                   flux_eqs=None, extra_eqs=(), diagram=None):
    """Decorator: register a dydt function as a named model.

    ``fluxes`` is an optional companion function returning a dict of internal
    fluxes for the same state; if given, :func:`flux_timeseries` can reconstruct
    the fluxes over a solution. The remaining keywords are documented on
    :class:`HLMModel`.
    """
    if default_params is not None and len(param_names) != len(default_params):
        raise ValueError(f"{name}: param_names and default_params differ in length")
    if y0 is not None and len(y0) != dim:
        raise ValueError(f"{name}: y0 must have length dim={dim}")

    def deco(fn):
        MODELS[name] = HLMModel(
            name=name, dydt=fn, dim=dim,
            state_names=tuple(state_names), description=description,
            fluxes=fluxes, precalc=precalc,
            param_names=tuple(param_names),
            default_params=None if default_params is None
                           else tuple(float(v) for v in default_params),
            y0=None if y0 is None else tuple(float(v) for v in y0),
            storages=tuple(storages), streamflow=streamflow,
            balance=balance, flux_eqs=flux_eqs, extra_eqs=tuple(extra_eqs),
            diagram=diagram,
        )
        return fn
    return deco


# Source / sink nodes of the flow diagram.
RAIN, ET, CHANNEL = "Rain", "ET", "Channel"


def flow_edges(model):
    """Flow-diagram edges ``[(flux_key, from, to), ...]`` derived from ``balance``.

    A flux leaving one storage and entering another links the two. A flux that
    only enters storages comes from ``RAIN``; one that only leaves goes to
    ``ET`` if its key starts with ``e`` (evaporation), else to ``CHANNEL``.
    ``model.diagram`` overrides single fluxes (``None`` hides one), for flows
    the balance alone cannot place, such as a flux split between two targets.
    """
    src, dst = {}, {}
    for storage, (ins, outs) in model.balance.items():
        for k in ins:
            dst[k] = storage
        for k in outs:
            src[k] = storage
    edges = {}
    for k in list(src) + [k for k in dst if k not in src]:
        default_sink = ET if k.startswith("e") else CHANNEL
        edges[k] = (src.get(k, RAIN), dst.get(k, default_sink))
    edges.update(model.diagram or {})
    return [(k, a, b) for k, ab in edges.items() if ab is not None
            for a, b in [ab]]


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


def simulate(model, L, A_h, A_i, forcing, t_end, global_params=None, y0=None,
             t_eval=None):
    """Run one single-link simulation of a runnable model.

    Parameters
    ----------
    model : HLMModel
    L, A_h, A_i : channel length [m], hillslope area [m^2], upstream area [km^2]
    forcing : ArrayForcing or 2-tuple (see RHS contract)
    t_end : float, simulation length [min]
    global_params, y0 : override the model's defaults if given
    t_eval : times [min] at which to store the solution

    Returns
    -------
    dict with ``t`` [min], ``y`` (dim, N), ``fluxes`` (dict of arrays) and
    ``q`` (streamflow [m^3/s], shape (N,)).
    """
    if not model.runnable:
        raise ValueError(f"model {model.name!r} is missing simulation metadata")
    gp = np.array(model.default_params if global_params is None else global_params,
                  dtype=float)
    y0 = np.array(model.y0 if y0 is None else y0, dtype=float)
    params = model.precalc(L, A_h, A_i, gp)

    sol = solve_ivp(model.dydt, (0.0, t_end), y0, args=(params, gp, forcing),
                    method="LSODA", rtol=1e-6, atol=1e-9, max_step=30.0,
                    t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(f"{model.name}: {sol.message}")

    f = flux_timeseries(model, sol.t, sol.y, params, gp, forcing)
    if model.streamflow in model.state_names:
        q = sol.y[model.state_names.index(model.streamflow)]
    else:
        q = f[model.streamflow]
    return {"t": sol.t, "y": sol.y, "fluxes": f, "q": q}


# (name, unit) of the 254-family global parameters, in index order.
PARAM_NAMES_254 = (
    ("v_0", "m/s"), ("lambda_1", "-"), ("lambda_2", "-"), ("v_h", "m/s"),
    ("k_3", "1/min"), ("k_I_factor", "-"), ("h_b", "m"), ("S_L", "m"),
    ("A", "-"), ("B", "-"), ("exponent", "-"), ("v_B", "m/s"),
)
PARAM_NAMES_255 = PARAM_NAMES_254 + (
    ("Ssmax", "m"), ("Ssmin", "m"), ("Ss_coef", "1/m"), ("Ss_exp", "-"),
)
Y0_254 = (0.001, 0.0, 0.05, 0.20, 0.0, 0.0, 0.001)
STORAGES_254 = ("s_p", "s_t", "s_s")

# Equations of the 254 family (see HLMModel.balance / flux_eqs / extra_eqs).
BALANCE_254 = {
    "s_p": (("rain_in",), ("q_pl", "q_pt", "e_p")),
    "s_t": (("q_pt",),    ("q_ts", "e_t")),
    "s_s": (("q_ts",),    ("q_sl", "e_s")),
}
FLUX_EQS_254 = {
    "rain_in": (r"P",      r"c_1\,p(t)"),
    "e_p":     (r"E_p",    r"\frac{S_p}{C}\,E_{pot}"),
    "e_t":     (r"E_t",    r"\frac{S_t / S_L}{C}\,E_{pot}"),
    "e_s":     (r"E_s",    r"\frac{S_s / (h_b - S_L)}{C}\,E_{pot}"),
    "k_t":     (r"k_t",    r"k_2\left(A + B\left(1 - \frac{S_t}{S_L}\right)^{\text{exponent}}\right)"),
    "q_pl":    (r"q_{pl}", r"k_2\,S_p"),
    "q_pt":    (r"q_{pt}", r"k_t\,S_p"),
    "q_ts":    (r"q_{ts}", r"k_i\,S_t"),
    "q_sl":    (r"q_{sl}", r"k_3\,S_s"),
}
EXTRA_EQS_254 = (
    (r"C",          r"S_p + \frac{S_t}{S_L} + \frac{S_s}{h_b - S_L}"),
    (r"k_2",        r"60\,v_h\,\frac{L}{A_h}"),
    (r"k_i",        r"k_{I,\text{factor}}\;k_2"),
    (r"\tau^{-1}",  r"\frac{60\,v_0\,A^{\lambda_2}}{(1 - \lambda_1)\,L}"),
    (r"\frac{dq}{dt}",
     r"\tau^{-1} q^{\lambda_1}\left(-q + c_2\,(q_{pl} + q_{sl}) + q_{up}\right)"),
    (r"\frac{dq_b}{dt}",
     r"\frac{v_B}{L}\left(A_h\,q_{sl} - 60\,q_b + 60\,q_{b,up}\right)"),
    (r"c_1,\ c_2", r"\frac{10^{-3}}{60}\ \text{(mm/h} \to \text{m/min)},\quad \frac{A_h}{60}\ \text{(m/min} \to \text{m}^3\text{/s)}"),
)


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
    precalc=precalculations, param_names=PARAM_NAMES_254,
    default_params=GLOBAL_PARAMS_254, y0=Y0_254,
    storages=STORAGES_254, streamflow="q",
    balance=BALANCE_254, flux_eqs=FLUX_EQS_254, extra_eqs=EXTRA_EQS_254,
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
    precalc=precalculations, param_names=PARAM_NAMES_254,
    default_params=GLOBAL_PARAMS_254, y0=Y0_254,
    storages=STORAGES_254, streamflow="q",
    balance=BALANCE_254,
    flux_eqs={**FLUX_EQS_254, "k_t": (r"k_t", r"A\,k_2")},
    extra_eqs=EXTRA_EQS_254,
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
    precalc=precalculations, param_names=PARAM_NAMES_255,
    default_params=GLOBAL_PARAMS_255, y0=Y0_254,
    storages=STORAGES_254, streamflow="q",
    balance=BALANCE_254,
    flux_eqs={**FLUX_EQS_254, "q_sl": (r"q_{sl}", r"""\begin{cases}
        0 & S_s \le S_{s,min} \\
        k_3\,S_s & S_{s,min} < S_s \le S_{s,max} \\
        k_3\,S_s + c_s\,(S_s - S_{s,max})\,e^{\eta_s (S_s - S_{s,max})} & S_s > S_{s,max}
        \end{cases}""")},
    extra_eqs=EXTRA_EQS_254 + (
        (r"c_s,\ \eta_s", r"\text{Ss\_coef},\ \text{Ss\_exp}"),),
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


SMOOTH_EXCESS_EQ = (
    r"\sigma(S, S_{max})",
    r"\left[1 + \exp\left(-\frac{S - S_{max} + r\,e\,S_{max}}{r\,S_{max}}\right)\right]^{-1}"
    rf",\quad r = {CRB3_SMOOTH_R:g},\ e = {CRB3_SMOOTH_E:g}",
)


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
    precalc=precalculations_crb3,
    param_names=(("A", "m^(1-B)/min"), ("B", "-"), ("lambda", "-"),
                 ("Ssmax", "m"), ("Sfc", "-"), ("M", "-"),
                 ("A2", "m^(1-B2)/min"), ("B2", "-"), ("v_h", "m/s")),
    # setup defaults with the notebook's non-stiff A and B2
    default_params=np.r_[1e-3, GLOBAL_PARAMS_CRB3[1:7], 1.0, GLOBAL_PARAMS_CRB3[8]],
    y0=(0.0, 0.05, 0.0, 0.001),
    storages=("s_s", "s_p", "s_g"), streamflow="q_out",
    balance={
        "s_s": (("rain_in",), ("e_v", "e_b", "q_sp", "q_ss")),
        "s_p": (("q_sp",),    ("q_pL",)),
        "s_g": (("q_sg",),    ("q_gL",)),
    },
    flux_eqs={
        "rain_in": (r"P",       r"c_1\,p(t)"),
        "e_v":     (r"E_v",     r"\begin{cases} M\,E_{pot} & S_s > S_{fc} \\ \frac{S_s}{S_{fc}}\,M\,E_{pot} & \text{otherwise} \end{cases}"),
        "e_b":     (r"E_b",     r"\frac{S_s}{S_{s,max}}\,(1 - M)\,E_{pot}"),
        "q_sp":    (r"q_{sp}",  r"P\,\sigma(S_s, S_{s,max})"),
        "q_ss":    (r"q_{ss}",  r"\begin{cases} A\,(S_s - S_{fc})^{B} & S_s > S_{fc} \\ 0 & \text{otherwise} \end{cases}"),
        "q_sg":    (r"q_{sg}",  r"\lambda\,q_{ss}"),
        "q_sL":    (r"q_{sL}",  r"(1 - \lambda)\,q_{ss}"),
        "q_pL":    (r"q_{pL}",  r"k_2\,S_p"),
        "q_gL":    (r"q_{gL}",  r"A_2\,S_g^{B_2}"),
        "q_out":   (r"Q_{out}", r"c_2\,(q_{pL} + q_{gL} + q_{sL})"),
    },
    extra_eqs=(
        (r"S_{fc}", r"\text{Sfc}\cdot S_{s,max}"),
        SMOOTH_EXCESS_EQ,
        (r"k_2", r"60\,v_h\,\frac{L}{A_h}"),
        (r"\frac{dQ}{dt}", r"q_{pL} + q_{gL} + q_{sL}\quad \text{(cumulative runoff)}"),
        (r"c_1,\ c_2", r"\frac{10^{-3}}{60}\ \text{(mm/h} \to \text{m/min)},\quad \frac{A_h}{60}\ \text{(m/min} \to \text{m}^3\text{/s)}"),
    ),
    # q_ss leaves Ss but is split: q_sg -> Sg and q_sL -> channel.
    diagram={"q_ss": None, "q_sg": ("s_s", "s_g"), "q_sL": ("s_s", CHANNEL)},
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


# ----------------------------------------------------------------------------
# New Zealand V1 -- soil-matrix (Sm) / ponded (Sp) storages
# ----------------------------------------------------------------------------
def fluxes_nzv1(t, y, params, global_params, forcing):
    S_fc  = global_params[0]        # [m]
    S_max = global_params[1]        # [m]
    V_p   = global_params[2]        # [m/s]
    V_s   = global_params[3]        # [m/s]
    V_b   = global_params[4]        # [m/s]
    a_p   = global_params[5]
    a_s   = global_params[6]
    a_b   = global_params[7]
    M     = global_params[8]

    rain_mm_hr, e_pot = _read_forcing(forcing, t)
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params
    s_m, s_p = max(y[1], 0.0), max(y[2], 0.0)

    P = rain_mm_hr * c_1

    # release coefficients from hillslope geometry
    K_p = L / A_h * V_p * 60.0
    K_s = L / A_h * V_s * 60.0
    K_b = L / A_h * V_b * 60.0

    # evaporation: vegetation and bare soil
    e_v = M * e_pot if s_m > S_fc else s_m / S_fc * M * e_pot
    e_b = s_m / S_max * (1.0 - M) * e_pot

    # saturation excess -> ponded storage
    q_mp = P * _smooth_excess(s_m, S_max)

    q_pL = K_p * s_p ** a_p
    q_sL = K_s * (s_m - S_fc) ** a_s if s_m > S_fc else 0.0
    q_bL = K_b * s_m ** a_b

    return {
        "rain_in": P,
        "e_v": e_v, "e_b": e_b,
        "q_mp": q_mp, "q_pL": q_pL, "q_sL": q_sL, "q_bL": q_bL,
        "q_out": (q_pL + q_sL + q_bL) * c_2,    # hillslope outflow [m^3/s]
    }


@register_model(
    "NewZealandV1", dim=3,
    state_names=("Q", "s_m", "s_p"),
    description="New Zealand V1: soil-matrix (Sm) and ponded (Sp) storages. "
                "Q is cumulative runoff [m]; streamflow is the 'q_out' flux "
                "[m^3/s].",
    fluxes=fluxes_nzv1,
    precalc=precalculations_nzv1,
    param_names=(("S_fc", "m"), ("S_max", "m"), ("V_p", "m/s"),
                 ("V_s", "m/s"), ("V_b", "m/s"), ("a_p", "-"), ("a_s", "-"),
                 ("a_b", "-"), ("M", "-")),
    default_params=GLOBAL_PARAMS_NZV1, y0=(0.0, 0.05, 0.0),
    storages=("s_m", "s_p"), streamflow="q_out",
    balance={
        "s_m": (("rain_in",), ("e_v", "e_b", "q_mp", "q_sL", "q_bL")),
        "s_p": (("q_mp",),    ("q_pL",)),
    },
    flux_eqs={
        "rain_in": (r"P",       r"c_1\,p(t)"),
        "e_v":     (r"E_v",     r"\begin{cases} M\,E_{pot} & S_m > S_{fc} \\ \frac{S_m}{S_{fc}}\,M\,E_{pot} & \text{otherwise} \end{cases}"),
        "e_b":     (r"E_b",     r"\frac{S_m}{S_{max}}\,(1 - M)\,E_{pot}"),
        "q_mp":    (r"q_{mp}",  r"P\,\sigma(S_m, S_{max})"),
        "q_pL":    (r"q_{pL}",  r"K_p\,S_p^{a_p}"),
        "q_sL":    (r"q_{sL}",  r"\begin{cases} K_s\,(S_m - S_{fc})^{a_s} & S_m > S_{fc} \\ 0 & \text{otherwise} \end{cases}"),
        "q_bL":    (r"q_{bL}",  r"K_b\,S_m^{a_b}"),
        "q_out":   (r"Q_{out}", r"c_2\,(q_{pL} + q_{sL} + q_{bL})"),
    },
    extra_eqs=(
        (r"K_p,\ K_s,\ K_b", r"60\,\frac{L}{A_h}\,V_p,\quad 60\,\frac{L}{A_h}\,V_s,\quad 60\,\frac{L}{A_h}\,V_b"),
        SMOOTH_EXCESS_EQ,
        (r"\frac{dQ}{dt}", r"q_{pL} + q_{sL} + q_{bL}\quad \text{(cumulative runoff)}"),
        (r"c_1,\ c_2", r"\frac{10^{-3}}{60}\ \text{(mm/h} \to \text{m/min)},\quad \frac{A_h}{60}\ \text{(m/min} \to \text{m}^3\text{/s)}"),
    ),
)
def model_nzv1(t, y, params, global_params, forcing,
               parents_q=0.0, parents_qb=0.0):
    # parents_q / parents_qb are ignored: Q is not routed in this model.
    f = fluxes_nzv1(t, y, params, global_params, forcing)
    ans = np.zeros(3)
    ans[0] = f["q_pL"] + f["q_sL"] + f["q_bL"]                          # Q
    ans[1] = (f["rain_in"] - f["e_v"] - f["e_b"] - f["q_mp"]
              - f["q_sL"] - f["q_bL"])                                  # Sm
    ans[2] = f["q_mp"] - f["q_pL"]                                      # Sp
    return ans


# ----------------------------------------------------------------------------
# ConstantRunoff -- asynch 190 hillslope (UMN_HLM_Coupled model_ConstantRunoff)
# ----------------------------------------------------------------------------
# Port of runoff/src/models/model_ConstantRunoff.hpp. The C++ model computes
# PET from t2m with Oudin; here PET comes from the forcing like every other
# playground model.
def fluxes_constant_runoff(t, y, params, global_params, forcing):
    rc  = global_params[0]
    v_g = global_params[2]          # [m/s]

    rain_mm_hr, e_pot = _read_forcing(forcing, t)
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params
    s_p, s_a = max(y[0], 0.0), max(y[1], 0.0)

    P = rain_mm_hr * c_1
    k_3 = v_g * L / A_h * 60.0      # baseflow rate [1/min]

    # asynch 190: PET shared in proportion to storage
    e_p = e_a = 0.0
    if e_pot > 0.0:
        C_p = s_p / e_pot
        C_a = s_a / e_pot
        C_T = C_p + C_a
        corr = 1.0 / C_T if C_T > 1.0 else 1.0
        e_p = corr * C_p * e_pot
        e_a = corr * C_a * e_pot

    q_pl = k_2 * s_p
    q_al = k_3 * s_a

    return {
        "rain_in": P,
        "q_rp": rc * P, "q_ra": (1.0 - rc) * P,     # rain -> runoff / baseflow
        "e_p": e_p, "e_a": e_a,
        "q_pl": q_pl, "q_al": q_al,
        "q_out": (q_pl + q_al) * c_2,               # hillslope outflow [m^3/s]
    }


@register_model(
    "ConstantRunoff", dim=4,
    state_names=("s_p", "s_a", "surf_runoff", "total_runoff"),
    description="Constant runoff coefficient (asynch 190 hillslope): rain is "
                "split by rc between a fast runoff (s_p) and a slow baseflow "
                "(s_a) linear reservoir. Streamflow is the 'q_out' flux "
                "[m^3/s].",
    fluxes=fluxes_constant_runoff,
    precalc=precalculations_constant_runoff,
    param_names=(("rc", "-"), ("v_h", "m/s"), ("v_g", "m/s")),
    default_params=GLOBAL_PARAMS_CONSTANT_RUNOFF,
    y0=(0.0, 0.01, 0.0, 0.0),
    storages=("s_p", "s_a"), streamflow="q_out",
    balance={
        "s_p": (("q_rp",), ("q_pl", "e_p")),
        "s_a": (("q_ra",), ("q_al", "e_a")),
    },
    flux_eqs={
        "rain_in": (r"P",       r"c_1\,p(t)"),
        "q_rp":    (r"P_p",     r"r_c\,P"),
        "q_ra":    (r"P_a",     r"(1 - r_c)\,P"),
        "e_p":     (r"E_p",     r"S_p\,\min\left(1,\ \frac{E_{pot}}{S_p + S_a}\right)"),
        "e_a":     (r"E_a",     r"S_a\,\min\left(1,\ \frac{E_{pot}}{S_p + S_a}\right)"),
        "q_pl":    (r"q_{pl}",  r"k_2\,S_p"),
        "q_al":    (r"q_{al}",  r"k_3\,S_a"),
        "q_out":   (r"Q_{out}", r"c_2\,(q_{pl} + q_{al})"),
    },
    extra_eqs=(
        (r"k_2,\ k_3", r"60\,v_h\,\frac{L}{A_h},\quad 60\,v_g\,\frac{L}{A_h}"),
        (r"\frac{d\,\text{surf\_runoff}}{dt}", r"q_{pl}\quad \text{(cumulative)}"),
        (r"\frac{d\,\text{total\_runoff}}{dt}", r"q_{pl} + q_{al}\quad \text{(cumulative)}"),
        (r"c_1,\ c_2", r"\frac{10^{-3}}{60}\ \text{(mm/h} \to \text{m/min)},\quad \frac{A_h}{60}\ \text{(m/min} \to \text{m}^3\text{/s)}"),
    ),
)
def model_constant_runoff(t, y, params, global_params, forcing,
                          parents_q=0.0, parents_qb=0.0):
    # parents_q / parents_qb are ignored: routing is not part of this model.
    f = fluxes_constant_runoff(t, y, params, global_params, forcing)
    ans = np.zeros(4)
    ans[0] = f["q_rp"] - f["q_pl"] - f["e_p"]          # s_p
    ans[1] = f["q_ra"] - f["q_al"] - f["e_a"]          # s_a
    ans[2] = f["q_pl"]                                 # surf_runoff
    ans[3] = f["q_pl"] + f["q_al"]                     # total_runoff
    return ans
