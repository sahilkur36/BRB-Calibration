# Standalone Bayesian / quoFEM helper (CB225)



This folder is self-contained: run everything from here with Python 3 and OpenSeesPy installed (`pip install -r requirements.txt`).

**Units:** Python APIs assume one consistent system—length $[\mathrm{in}]$, force $[\mathrm{kip}]$, stress $[\mathrm{ksi}]$, area $[\mathrm{in}^2]$. `config/specimen_config.yaml` still uses suffix-style keys (`fy_ksi`, `L_T_in`, …) for readability; `lib/specimen_config.py` maps them into code.



## Layout



- **`config/specimen_config.yaml`** — material/geometry and paths (including where to write `cycle_meta`, targets, `landmark_cache.json`). Paths in `paths.*` are relative to the `config/` directory.

- **`data/force_deformation.csv`** — **resampled** experimental $P$–$\delta$ (columns `Deformation[in]`, `Force[kip]`). Row count must match $n$ in the cycle-points JSON.

- **`data/CB225_cycle_points.json`** — resampled cycle anchors (`zero_def`, etc.); $n$ equals the length of `force_deformation.csv`.

- **`config/cycle_meta.json`** — weight cycles and $w_c$ (**produced by `scripts/setup_cycle_targets.py`**).

- **`data/landmark_cache.json`** — frozen **characteristic-point** pairing on the shared displacement grid (**`scripts/precompute_landmark_cache.py`**). **`model.py` uses this for the likelihood vector and does not read `target_force.csv` for that step.** (The filename still says “landmark” for history.)

- **`target_displacement.csv`, `target_force.csv`** (bayesian root) — single-row comma-separated targets. **Produced by `scripts/setup_cycle_targets.py`**; used by precompute, `model.py` (displacement drive), and plotting.

- **`lib/`** — Python modules (characteristic-point / $J_{\mathrm{feat}}$ logic, specimen config, cycle I/O).

- **`scripts/`** — CLI utilities (**`clear.py`** resets generated outputs, then setup, precompute, calibration row, overlay plot).



Root artifacts used by the workflow / UQ: **`calibration_data.csv`**, **`results.out`**, **`predicted_force.csv`**, **`predicted_vs_calibration.png`** (when you run the plot script).



## Characteristic points and weights

Fit is measured at selected **(deformation, force)** locations $(\delta, P)$ on each hysteresis **cycle**, not by matching every resampled sample along the trace. **Characteristic points** include zero-force crossings, zero-deformation crossings, peak tensile and compressive force, and extra points along post-yield branches with $|P| > f_y A_{sc}$ to represent hardening.

For each paired point $k$, $(P_k^{\mathrm{num}}, \delta_k^{\mathrm{num}})$ and $(P_k^{\mathrm{exp}}, \delta_k^{\mathrm{exp}})$ are simulation and experiment. Deviations and global normalization scales from the full experimental history are

$$\Delta P_k = P_k^{\mathrm{num}} - P_k^{\mathrm{exp}}, \qquad \Delta \delta_k = \delta_k^{\mathrm{num}} - \delta_k^{\mathrm{exp}}$$

$$S_P = \max_i P_i^{\mathrm{exp}} - \min_i P_i^{\mathrm{exp}}, \qquad S_\delta = \max_i \delta_i^{\mathrm{exp}} - \min_i \delta_i^{\mathrm{exp}}.$$

Pointwise losses use the $L_2$ and $L_1$ forms

$$e_k^{(2)} = \left(\frac{\Delta P_k}{S_P}\right)^2 + \left(\frac{\Delta \delta_k}{S_\delta}\right)^2, \qquad e_k^{(1)} = \frac{|\Delta P_k|}{S_P} + \frac{|\Delta \delta_k|}{S_\delta}.$$

Let $\bar{e}_c^{(q)}$ be the mean of $e_k^{(q)}$ over all **valid** feature points in cycle $c$ ($q \in \{1,2\}$). The specimen-level characteristic-feature objective is the amplitude-weighted average over $N_c$ cycles,

$$J_{\mathrm{feat}}^{(q)} = \frac{\sum_{c=1}^{N_c} w_c \, \bar{e}_c^{(q)}}{\sum_{c=1}^{N_c} w_c}.$$

Cycle weights emphasize large-amplitude loops while keeping small cycles in the sum:

$$w_c = \left( \frac{\delta_c^{\max}}{\max_r \delta_r^{\max}} \right)^p + \varepsilon,$$

where $\delta_c^{\max}$ is the maximum absolute deformation reached in cycle $c$. With **`use_amplitude_weights: true`** (default), **`amplitude_weight_power`** and **`amplitude_weight_eps`** in `specimen_config.yaml` set $p$ and $\varepsilon$ (defaults $p = 2$, $\varepsilon = 0.05$). With **`use_amplitude_weights: false`**, use $w_c = 1$ for all cycles. **`config/cycle_meta.json`** stores $S_P$ and $S_\delta$ under the JSON keys **`s_f`** and **`s_d`**; $w_c$ is stored there per cycle.

**Implementation on the resampled grid.** Experimental data are a **resampled** polyline (`data/force_deformation.csv`). **Weight cycles** are index spans between zero-deformation anchors in the cycle JSON. On each span the code fills up to **twelve slots** (indices $0$–$11$; some filenames still say “landmark”) with the roles above: cycle start, peak-related anchors, signed tension/compression peaks, yield-region vertices with $|P| > f_y A_{sc}$ on the correct deflection side, $P = 0$ crossings at extremal deformation (segment-interpolated), and half-amplitude crossings toward peaks and yields. Cycles that never reach about $\pm 2 \delta_y$ are skipped ($\delta_y$ from steel and brace geometry). **`data/landmark_cache.json`** freezes the same displacement-grid indices for the OpenSees model. Slot logic lives in **`lib/jfeat_landmarks.py`** and **`lib/landmark_vector.py`**.

**Vector for quoFEM / TMCMC.** `calibration_data.csv` and `results.out` are one row of weighted normalized samples: for each contributing slot in cycle order, append

$$\sqrt{\frac{w_c}{n_c}}\,\frac{\delta}{S_\delta} \quad\text{and}\quad \sqrt{\frac{w_c}{n_c}}\,\frac{P}{S_P},$$

with $n_c$ the number of contributing slots in that cycle (so extra active slots in a cycle do not automatically overweight it). That row feeds **`defaultLogLikeScript.py`** as `calibrationData` and `prediction`.



## Workflow



### What runs once vs every TMCMC call vs optional QC

- **Run once per calibration problem** (repeat only if you change the experiment, resampled history, cycle JSON, `specimen_config.yaml`, or amplitude-weight settings): **steps 2–4** — `scripts/setup_cycle_targets.py`, `scripts/precompute_landmark_cache.py`, `scripts/write_weighted_calibration.py`. Together they fix **`cycle_meta.json`**, **`target_displacement.csv`** / **`target_force.csv`**, **`data/landmark_cache.json`**, and **`calibration_data.csv`** (the experimental characteristic-point row used as `calibrationData`).

- **Run on every TMCMC likelihood evaluation** (every proposed parameter vector $\boldsymbol{\theta}$): **`model.py`** with **`params.py`** set to that $\boldsymbol{\theta}$ (your UQ driver overwrites `params.py` or equivalent before each call). Same displacement drive and cache; output **`results.out`** is the model prediction row passed as `prediction` to **`defaultLogLikeScript.py`** next to **`calibration_data.csv`**. You do **not** rerun steps 2–4 inside the sampling loop unless inputs or $\boldsymbol{\theta}$-dependent geometry logic for the *experimental* side changes (here geometry in config is fixed; only material parameters in `params.py` change).

- **Optional** (not used by the likelihood): **step 6** — `scripts/plot_predicted_vs_calibration.py` for a $P$–$\delta$ overlay and **`predicted_force.csv`**. **Step 1** — `scripts/clear.py` is optional housekeeping before regenerating artifacts.



1. **Clear generated outputs** — *optional* — run before regenerating artifacts from scratch. Removes `cycle_meta`, target CSVs, characteristic-point cache (`landmark_cache.json`), `calibration_data.csv`, `results.out`, `predicted_force.csv`, and `predicted_vs_calibration.png`—paths taken from `config/specimen_config.yaml` where applicable. Does not delete `data/force_deformation.csv`, cycle JSON, or the config file. Use `--dry-run` to list files without deleting.



   ```bash

   python scripts/clear.py

   ```



2. **Setup: cycle meta + target CSVs** — *once per problem (or when inputs change)* — after editing `data/force_deformation.csv`, cycle JSON, `config/specimen_config.yaml`, or amplitude settings:



   ```bash

   python scripts/setup_cycle_targets.py

   ```



   Writes `config/cycle_meta.json`, `target_displacement.csv`, and `target_force.csv` (paths from `specimen_config.yaml` under `paths.*`).



3. **Characteristic-point cache** — *once per problem (or when targets / meta / specimen gates change)*:



   ```bash

   python scripts/precompute_landmark_cache.py

   ```



   Writes `data/landmark_cache.json` by default (override with `-o`). This freezes which grid indices pair experiment and model at each characteristic point so `model.py` does not need `target_force.csv` for `results.out`.



4. **Calibration row** — *once per problem* — weighted characteristic-point vector from the experiment only; order matches `results.out` (`calibrationData` for TMCMC):



   ```bash

   python scripts/write_weighted_calibration.py -o calibration_data.csv

   ```



5. **Forward model** — ***every TMCMC / likelihood call*** — runs OpenSees with **`params.py`** for the current $\boldsymbol{\theta}$, reads `data/landmark_cache.json`, writes **`results.out`** as the model prediction row (`prediction` for `defaultLogLikeScript.py`; not one value per raw history step):



   ```bash

   python model.py

   ```



   Requires `data/landmark_cache.json` (same layout as after `precompute_landmark_cache`).



6. **Overlay plot** — *optional QC / visualization* — runs the forward model, writes **`predicted_force.csv`**, then plots normalized $P$–$\delta$ (**top** axis deformation $[\mathrm{in}]$, **right** axis force $[\mathrm{kip}]$):



   ```bash

   python scripts/plot_predicted_vs_calibration.py -o predicted_vs_calibration.png

   ```



With identical simulated and experimental force, `results.out` matches `calibration_data.csv` (up to float noise).



## Order of values (`calibration_data.csv` / `results.out`)



Both files are a **single row** of the same length. Values are built by walking **`meta` in `cycle_meta.json`** (weight cycles, in file order). For each cycle:



1. **Slots** are the twelve $J_{\mathrm{feat}}$ characteristic-point indices ($0$–$11$). **Only slots that contribute** to the loss for that cycle are emitted; inactive slots are skipped (so length is not fixed at $12 \times 2$ per cycle).

2. For **each contributing slot**, two numbers are appended: $\sqrt{w_c / n_c}\,(\delta / S_\delta)$ then $\sqrt{w_c / n_c}\,(P / S_P)$. $w_c$ and $n_c$ are the per-cycle weight and contributing-count from **Characteristic points and weights** above; $S_P$ and $S_\delta$ are the global scales (`s_f` / `s_d` in `cycle_meta.json`). $\delta$ and $P$ are the paired coordinates at that characteristic point: experiment in `calibration_data.csv`, simulation in `results.out` using the same grid pairing from `landmark_cache.json`. See `lib/landmark_vector.py`.



Layout: $\mathrm{cycle}_0 \to$ contributing slots in index order $\to$ (weighted $\hat{\delta}$, weighted $\hat{P}$) each $\to$ $\mathrm{cycle}_1 \to \cdots$.



## Likelihood



`defaultLogLikeScript.py` expects `prediction` and `calibrationData` aligned with these characteristic-point vectors; apply the usual shift/scale if your UQ workflow requires it.

