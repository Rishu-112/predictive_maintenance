# Predictive Maintenance of Turbofan Engines — RUL Prediction

**Role target:** Data Scientist / Machine Learning Engineer (campus placements)  
**Methods:** regression, time-series feature engineering, engine-level validation, explainable evaluation, Streamlit

## Goal
Predict an engine's **remaining useful life (RUL), measured in operating cycles**, using its observations up to the present cycle. Lower RUL indicates a shorter simulated period until engine failure; predictions are **not** operational safety advice.

## Data
Use the **NASA C-MAPSS Turbofan Engine Degradation Simulation dataset**, subset **FD001** (100 simulated run-to-failure training engines, 100 truncated test engines). Dataset information: https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data and NASA repository https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Download the C-MAPSS dataset from the NASA repository, extract it, and copy these three files into `data/raw/`:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

**Do not confuse this dataset with N-CMAPSS, which has a different file format.** No actual NASA data is included in this ZIP. The `--demo` flag produces **artificial generated observations** solely to demonstrate the code.

## Windows PowerShell setup

```powershell
cd "path\to\turbofan_rul_project"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, you can run `\.venv\Scripts\python.exe` instead of `python` directly.

### Fast smoke test without downloading data

```powershell
python -m src.train --demo
streamlit run app.py
```

**Important:** the generated metrics in demo mode are *not* NASA scores and must not be listed on a résumé.

### Train and evaluate on official NASA FD001

```powershell
python -m src.train
streamlit run app.py
```

### Predict from a new NASA-format sequence file

```powershell
python -m src.predict --input data/raw/test_FD001.txt
```

### Run automated tests

```powershell
python -m pytest -q
```

## What the project actually does

1. Reads each engine's ordered 26-column time series (engine id, cycle, 3 settings, 21 sensors).
2. Creates training RUL = final observed cycle - current cycle (training engines terminate at failure).
3. Creates official test targets using published RUL labels; only **one final observation per test engine** is used for official test scoring.
4. Engineers 5-cycle rolling means and standard deviations **using only the past and present** for a selected set of sensors.
5. Splits training data by **engine ID**, preventing rows of the same engine appearing in both train and validation.
6. Benchmarks median regression, Random Forest, and histogram gradient boosting; selects model with lowest validation RMSE at a randomly selected pre-failure checkpoint per validation engine.
7. Refits the chosen model using all training engines and evaluates once on the official held-out test engines.
8. Saves serialized fitted model, evaluation JSON, test predictions CSV, scatter plot, and Streamlit explorer.

## Model metrics

- **MAE:** average absolute prediction error (cycles)
- **RMSE:** penalizes large RUL prediction mistakes (cycles)
- **R²:** regression goodness of fit (can be negative)

**No performance numbers are claimed here before running real NASA data.** Official test results appear under `reports/metrics.json` after training.

## Repository structure

```text
src/data.py       # NASA loading, target construction, rolling features
src/train.py      # Engine-disjoint validation, model selection and test evaluation
src/predict.py    # CLI prediction for new sensor trajectories
app.py            # Streamlit dashboard
requirements.txt  # Dependencies
tests/            # Feature correctness and leakage tests
data/raw/         # Put official NASA files here (ignored by Git)
artifacts/        # Saved model (ignored by Git)
reports/          # Generated results (ignored by Git)
```

## Important methodology and limitations

- Measurements are from **simulated** engines, not actual aircraft maintenance systems.
- In real data, the identity and value of the sensors and operating regimes matter; FD001 has a single operating condition and a single fault type.
- The model includes cumulative operating cycle as an input. This is available at prediction time; it is not a failure label. Investigate its importance and generalization to different engine populations.
- The test set is never used to select the model. Validation emulates test truncation using one randomly chosen pre-failure checkpoint per held-out training engine.
- No future sensor readings, final engine lifetime, or actual RUL are used as model inputs.
- Training RUL is uncapped here; alternative capped-target approaches need careful justification and evaluation.
- The Streamlit dashboard visualizes evaluation results; it is **not** a live monitoring service.
- Before any GitHub publication: include real model results and your own exploratory analysis, document failure cases and limitations, and do not claim production deployment.

## Interview questions to prepare

1. Why must you split by engine instead of randomly by row?
2. What is RUL, and why is this a regression task?
3. Why compute rolling means and standard deviations?
4. How is the official test RUL constructed from the truncated observations?
5. Why evaluate test engines only at their last recorded cycle?
6. When would MAE be preferable to RMSE?
7. What makes this a predictive-maintenance rather than a standard tabular ML problem?
8. What sources of data leakage arise in temporal sensor models?
9. How could you add uncertainty intervals or maintenance cost-based evaluation?
10. What would need to change before the model could be used on real equipment?

## Résumé bullet template (fill verified results only)

> Developed an engine-level remaining useful life prediction pipeline on NASA C-MAPSS FD001 using causal rolling sensor features, Random Forest / Gradient Boosting benchmarks, and leakage-safe group validation; achieved **[verified test RMSE]** cycles RMSE on 100 held-out simulated engine trajectories.

> Built a Streamlit evaluation dashboard and reusable prediction CLI with automated tests and reproducible model artifacts.
