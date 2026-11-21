# Interview-ready project guide

## 30-second explanation
This project estimates remaining operating cycles before failure for simulated turbofan engines. I used NASA C-MAPSS FD001, created a remaining-useful-life target, engineered causal rolling features from sensor readings, compared multiple regression baselines with engine-disjoint validation, and evaluated a selected model on unseen engines. I built a dashboard to inspect predicted versus actual lifetimes.

## Why this works for a Data Scientist placement
This is a **supervised regression** task with **multivariate time series**. It demonstrates the importance of correct evaluation more than a flashy model: the same engine never appears in both training and validation subsets, and the final NASA test set is unseen during model selection.

## Explain the target carefully
Training engine trajectories go to simulated failure, so at cycle c of an engine with last cycle T, RUL = T - c. Official test trajectories stop before failure; the provided per-engine RUL label gives the remaining cycles after the final observed test cycle.

## Follow-up improvements you can implement
- Add per-engine EDA plots, sensor trend analysis, and feature importance / permutation importance.
- Compare early-cycle vs late-cycle error; examine absolute error by true RUL range.
- Test a limited max-RUL target transformation, but keep final reported errors on the original target scale.
- Try a temporal architecture (LSTM/GRU) only after a strong honest tabular baseline.
- Add a confidence interval or risk threshold and optimize it with maintenance cost assumptions.

## Do not claim
- Use on real airplanes
- NASA accuracy from `--demo`
- A production deployment
- State-of-the-art performance without fair published benchmarking
