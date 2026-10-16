# HLM models playground

A small Python sandbox for prototyping and comparing **hillslope-link
hydrological models** (the Hillslope Link Model family run by
[ASYNCH](https://github.com/Iowa-Flood-Center/asynch)) on a single link,
before committing to an implementation in the C solver.

Every model is a Python right-hand side that `scipy.integrate.solve_ivp` can
integrate. Each model also exposes its **internal fluxes** (infiltration,
percolation, runoff, evaporation, ...), so after a run you can plot every
flux, check that the water balance closes, and run parameter-sensitivity
sweeps, all from the same equations the solver integrated.

## Contents

| File | What it is |
|---|---|
| [`models.py`](models.py) | Model registry and the right-hand sides of all models (`254_toplayer`, `254_const_infil`, `255_NonLinearSub`, `CRB3`), plus `flux_timeseries`. |
| [`setup.py`](setup.py) | Model-independent helpers: `precalculations` (per-link parameters, as ASYNCH computes them), `ArrayForcing` (hourly rain/evap forcing) and default global-parameter vectors. |
| [`setup_Playground_HLM_CollieRB3.ipynb`](setup_Playground_HLM_CollieRB3.ipynb) | Single-link runs of model 255 and CRB3: hydrograph, storages, fluxes, water balance, one-at-a-time sensitivity and interactive hydrograph sweeps. |
| [`docs/adding_a_model.md`](docs/adding_a_model.md) | **How to add a new model** to the registry. |
| [`docs/flux_analysis.md`](docs/flux_analysis.md) | **How to run the flux analysis** (flux plots, water balance, sensitivity) for any model, including a new one. |

> `setup.py` is a plain helper module, not a packaging script. Don't run
> `pip install .` on it.

## Quick start

```bash
git clone <this repo> && cd hlm-models-playground
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab setup_Playground_HLM_CollieRB3.ipynb
```

Minimal run from Python:

```python
import numpy as np
from scipy.integrate import solve_ivp
from models import get_model, flux_timeseries
from setup import precalculations, ArrayForcing, GLOBAL_PARAMS_255

model = get_model("255_NonLinearSub")
gp = GLOBAL_PARAMS_255.copy()
params = precalculations(L=350.0, A_h=133356.8, A_i=50.0, global_params=gp)

rain = np.zeros(240); rain[11:17] = [5, 10, 17, 10, 8, 3]    # mm/hr, hourly
forcing = ArrayForcing(rain, epot=50.0, dt=60.0)              # epot in mm/month, dt in minutes

y0 = [0.001, 0.0, 0.05, 0.20, 0.0, 0.0, 0.001]                # see model.state_names
sol = solve_ivp(model.dydt, (0.0, forcing.t_end), y0,
                args=(params, gp, forcing), method="LSODA",
                rtol=1e-6, atol=1e-9, max_step=30.0, dense_output=True)

fluxes = flux_timeseries(model, sol.t, sol.y, params, gp, forcing)
print(model.state_names, list(fluxes))
```

## Conventions

- **Time** is in **minutes**. Forcing records are piecewise constant over
  `dt` (60 min for hourly data), as in ASYNCH.
- **Storages** are in **m**. **Hillslope fluxes** are in **m/min**. Channel
  discharge is in **m³/s**.
- **Rain** is given in mm/hr and **potential evaporation** in mm/month.
  `_read_forcing` converts both.
- **Per-link parameters** come from a `precalculations*` function and always
  have the layout `[L, A_h, invtau, k_2, k_i, c_1, c_2]`.

## Available models

| Name | States | Summary |
|---|---|---|
| `254_toplayer` | `q, s_p, s_t, s_s, s_precip, V_r, q_b` | ASYNCH model 254: ponded / top / sub-surface storages, saturation-excess infiltration. |
| `254_const_infil` | same as 254 | 254 with constant infiltration `k_t = A·k_2`. |
| `255_NonLinearSub` | same as 254 | 254 with a non-linear sub-surface → channel release (`Ssmin`, `Ssmax`, `Ss_coef`, `Ss_exp`). |
| `CRB3` | `Q, s_s, s_p, s_g` | Collie River Basin 3 (MARRMoT-style): soil, ponded and groundwater storages. No channel routing; streamflow is the `q_out` flux. |

To list them at runtime: `from models import MODELS; print(MODELS.keys())`.
