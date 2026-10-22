"""Shared setup helpers for HLM testing.

Contains the things that are *independent of the specific model RHS*:

* :func:`precalculations` -- reproduces Asynch's per-link parameter derivation
  (``invtau, k_2, k_i, c_1, c_2``) so you can feed raw ``.prm`` values.
* :class:`ArrayForcing`   -- piecewise-constant forcing driven by per-record
  arrays (e.g. one record per hour).
* ``GLOBAL_PARAMS_254``, ``GLOBAL_PARAMS_255``, ``GLOBAL_PARAMS_CRB3`` --
  default global-parameter vectors.

The model right-hand sides themselves live in ``models.py``.

Global-parameter vector layout (model 254 convention)
-----------------------------------------------------
    [0]  v_0          channel reference velocity   [m/s]
    [1]  lambda_1     non-linearity exponent on q
    [2]  lambda_2     area exponent (used in invtau)
    [3]  v_h          hillslope velocity           [m/s]
    [4]  k_3          sub-surface -> channel rate  [1/min]
    [5]  k_I_factor   k_i = k_I_factor * k_2
    [6]  h_b          total active soil depth      [m]
    [7]  S_L          top-layer depth              [m]
    [8]  A
    [9]  B
    [10] exponent
    [11] v_B          base-flow velocity           [m/s]
    [12] Ssmax                                      [m]
    [13] Ssmin                                      [m]
    [14] Ss_coef                                    [1/m]
    [15] Ss_exp                                     [-]

Global-parameter vector layout (Collie River Basin 3, ``GLOBAL_PARAMS_CRB3``)
---------------------------------------------------------------------------
    [0]  A        coef qss = A*(Ss - Sfc*Ssmax)^B    [m^(1-B)/min]
    [1]  B        exp qss                            [-]
    [2]  lambda   fraction of qss going to Sg        [-]
    [3]  Ssmax    max soil storage                   [m]
    [4]  Sfc      field capacity as fraction of Ssmax [-]
    [5]  M        forest fraction                    [-]
    [6]  A2       coef qgL = A2*Sg^B2                [m^(1-B2)/min]
    [7]  B2       exp qgL                            [-]
    [8]  v_h      hillslope velocity (for k_2)       [m/s]
Per-link params come from :func:`precalculations_crb3`.

Global-parameter vector layout (New Zealand V1, ``GLOBAL_PARAMS_NZV1``)
-----------------------------------------------------------------------
    [0]  S_fc     field capacity                     [m]
    [1]  S_max    max soil-matrix storage            [m]
    [2]  V_p      ponded velocity, K_p = L/A_h*V_p*60 [m/s]
    [3]  V_s      sub-surface velocity (K_s)         [m/s]
    [4]  V_b      base-flow velocity (K_b)           [m/s]
    [5]  a_p      exp qpL = K_p*Sp^a_p               [-]
    [6]  a_s      exp qsL = K_s*(Sm - S_fc)^a_s      [-]
    [7]  a_b      exp qbL = K_b*Sm^a_b               [-]
    [8]  M        forest fraction                    [-]
Per-link params come from :func:`precalculations_nzv1`.

Global-parameter vector layout (ConstantRunoff, ``GLOBAL_PARAMS_CONSTANT_RUNOFF``)
-------------------------------------------------------------------------------
    [0]  rc       runoff coefficient (rain -> fast)  [-]
    [1]  v_h      runoff reference velocity (k_2)    [m/s]
    [2]  v_g      baseflow reference velocity (k_3)  [m/s]
Per-link params come from :func:`precalculations_constant_runoff`.
"""

import numpy as np


# Default global parameters (typical operational model-254 values).
# Edit / copy these to test alternative configurations.
GLOBAL_PARAMS_254 = np.array([
    0.33,        # 0  v_0        [m/s]
    0.20,        # 1  lambda_1
    -0.10,       # 2  lambda_2
    0.02,        # 3  v_h        [m/s]
    2.0425e-6,   # 4  k_3        [1/min]
    0.02,        # 5  k_I_factor
    0.50,        # 6  h_b        [m]
    0.10,        # 7  S_L        [m]
    0.0,         # 8  A
    99.0,        # 9  B
    3.0,         # 10 exponent
    0.75,        # 11 v_B        [m/s]
])

GLOBAL_PARAMS_255 = np.array([
    0.33,        # 0  v_0        [m/s]
    0.20,        # 1  lambda_1
    -0.10,       # 2  lambda_2
    0.02,        # 3  v_h        [m/s]
    2.0425e-6,   # 4  k_3        [1/min]
    0.02,        # 5  k_I_factor
    0.50,        # 6  h_b        [m]
    0.10,        # 7  S_L        [m]
    0.0,         # 8  A
    99.0,        # 9  B
    3.0,         # 10 exponent
    0.75,        # 11 v_B        [m/s]
    0.20,        # 12 SsMax      [m]
    0.05,        # 13 SsMin      [m]
    2.0425e-6,   # 14 Ss_Coeff   [1/m]
    17,         # 15 Ss_exponent [-]
])

GLOBAL_PARAMS_CRB3 = np.array([
    0.1,          # 0 A       coef qss = A(Ss-Sfc)^B     [m^(1-B)/min]
    1.0,          # 1 B       exp qss                    [-]
    0.5,          # 2 lambda  fraction of qss to Sg      [-]
    0.15,         # 3 Ssmax   max soil storage           [m]
    0.20,         # 4 Sfc     field capacity / Ssmax     [-]
    0.1,          # 5 M       forest fraction            [-]
    0.003,        # 6 A2      coef qgL = A2*Sg^B2        [m^(1-B2)/min]
    0.20,         # 7 B2      exp qgL                    [-]
    0.02,         # 8 v_h     hillslope velocity (k_2)   [m/s]
])

GLOBAL_PARAMS_NZV1 = np.array([
    0.03,         # 0 S_fc    field capacity             [m]
    0.15,         # 1 S_max   max soil-matrix storage    [m]
    0.02,         # 2 V_p     ponded velocity            [m/s]
    0.002,        # 3 V_s     sub-surface velocity       [m/s]
    0.0002,       # 4 V_b     base-flow velocity         [m/s]
    1.0,          # 5 a_p     exp qpL                    [-]
    1.0,          # 6 a_s     exp qsL                    [-]
    1.0,          # 7 a_b     exp qbL                    [-]
    0.1,          # 8 M       forest fraction            [-]
])

# Defaults from UMN_HLM_Coupled (qgis_plugin runoff_models.py / Stony Brook).
GLOBAL_PARAMS_CONSTANT_RUNOFF = np.array([
    0.33,         # 0 rc      runoff coefficient         [-]
    0.1,          # 1 v_h     runoff velocity (k_2)      [m/s]
    2.2917e-5,    # 2 v_g     baseflow velocity (k_3)    [m/s]
])


def precalculations(L, A_h, A_i, global_params):
    """Reproduce Asynch's model-254 per-link parameter derivation.

    Asynch stores raw params ``[L, A_h, A_i]`` per link and then computes
    ``invtau, k_2, k_i, c_1, c_2`` in ``Precalculations``. This mirrors that so
    you can feed real ``.prm`` values directly.

    Parameters
    ----------
    L   : channel length              [m]
    A_h : hillslope area              [m^2]
    A_i : total upstream area         [km^2]   (drainage area at the link)
    global_params : sequence          global parameter vector

    Returns
    -------
    numpy.ndarray, shape (7,) -> [L, A_h, invtau, k_2, k_i, c_1, c_2]
    """
    v_0        = global_params[0]
    lambda_1   = global_params[1]
    lambda_2   = global_params[2]
    v_h        = global_params[3]
    k_I_factor = global_params[5]

    invtau = 60.0 * v_0 * (A_i ** lambda_2) / ((1.0 - lambda_1) * L)   # [1/min]
    k_2    = v_h * L / A_h * 60.0                                       # [1/min]
    k_i    = k_2 * k_I_factor                                           # [1/min]
    c_1    = 0.001 / 60.0          # rainfall mm/hr -> m/min
    c_2    = A_h / 60.0            # m/min * A_h -> m^3/s

    return np.array([L, A_h, invtau, k_2, k_i, c_1, c_2])


def precalculations_crb3(L, A_h, A_i, global_params):
    """Per-link parameters for the Collie River Basin 3 model.

    Same output layout as :func:`precalculations` so models share one
    ``params`` contract. CRB3 has no channel routing and no top->sub-surface
    split, so ``invtau`` and ``k_i`` are NaN (unused). ``A_i`` is accepted
    for signature parity only.

    Returns
    -------
    numpy.ndarray, shape (7,) -> [L, A_h, nan, k_2, nan, c_1, c_2]
    """
    v_h = global_params[8]

    k_2 = v_h * L / A_h * 60.0     # Sp -> channel rate  [1/min]
    c_1 = 0.001 / 60.0             # rainfall mm/hr -> m/min
    c_2 = A_h / 60.0                       # m/min * A_h -> m^3/s

    return np.array([L, A_h, np.nan, k_2, np.nan, c_1, c_2])


def precalculations_nzv1(L, A_h, A_i, global_params):
    """Per-link parameters for the New Zealand V1 model.

    Same output layout as :func:`precalculations`. NZV1 builds its own
    K_p, K_s, K_b from ``L``, ``A_h`` and the velocities in ``global_params``,
    so ``invtau``, ``k_2`` and ``k_i`` are NaN (unused). ``A_i`` is accepted
    for signature parity only.

    Returns
    -------
    numpy.ndarray, shape (7,) -> [L, A_h, nan, nan, nan, c_1, c_2]
    """
    c_1 = 0.001 / 60.0             # rainfall mm/hr -> m/min
    c_2 = A_h / 60.0               # m/min * A_h -> m^3/s

    return np.array([L, A_h, np.nan, np.nan, np.nan, c_1, c_2])


def precalculations_constant_runoff(L, A_h, A_i, global_params):
    """Per-link parameters for the ConstantRunoff model (asynch 190 hillslope).

    Same output layout as :func:`precalculations`. ``k_2`` is the runoff
    (fast) storage rate; the baseflow rate ``k_3`` is built in the flux
    function from ``v_g``. No channel routing, so ``invtau`` and ``k_i`` are
    NaN. ``A_i`` is accepted for signature parity only.

    Returns
    -------
    numpy.ndarray, shape (7,) -> [L, A_h, nan, k_2, nan, c_1, c_2]
    """
    v_h = global_params[1]

    k_2 = v_h * L / A_h * 60.0     # s_p -> channel rate  [1/min]
    c_1 = 0.001 / 60.0             # rainfall mm/hr -> m/min
    c_2 = A_h / 60.0               # m/min * A_h -> m^3/s

    return np.array([L, A_h, np.nan, k_2, np.nan, c_1, c_2])


class ArrayForcing:
    """Piecewise-constant forcing driven by per-record arrays.

    Mirrors Asynch's treatment of forcings: each record (e.g. one hour) holds
    its value constant over the whole interval ``dt``. Times before the first
    record use record 0; times at/after the last record hold the final value.

    Parameters
    ----------
    rain : 1-D array_like
        Rainfall per record [mm/hr].
    epot : float or 1-D array_like
        Potential evap [mm/month]. A scalar is broadcast to every record;
        an array must match ``len(rain)``.
    dt : float
        Record length in MINUTES (60.0 for hourly records).
    """

    def __init__(self, rain, epot, dt=60.0):
        self.rain = np.asarray(rain, dtype=float)
        self.n = self.rain.size
        if np.ndim(epot) == 0:
            self.epot = np.full(self.n, float(epot))
        else:
            self.epot = np.asarray(epot, dtype=float)
            if self.epot.size != self.n:
                raise ValueError("epot array must match rain length")
        self.dt = float(dt)

    def __call__(self, t):
        i = int(t // self.dt)                 # which record contains time t
        i = 0 if i < 0 else (self.n - 1 if i >= self.n else i)
        return self.rain[i], self.epot[i]

    @property
    def t_end(self):
        """End of the forcing series [min]."""
        return self.n * self.dt
