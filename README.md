# Offshore wave height (Hm0) prediction: Random Forest vs U-Net

Predicts significant wave height (Hm0) around Ireland (50–57°N, 15–4°W, 0.25° grid, hourly, ERA5 2005) from 10 m wind (u10, v10) and mean sea-level pressure (MSLP).

## Run

```bash
uv sync                                   # Python 3.13; installs numpy, pandas, scikit-learn, torch, shap, ...
unzip "Copy of Dataset and code - Wave Power.zip" -d "Copy of Dataset and code - Wave Power"
uv run --with jupyterlab jupyter lab src/off_shore/   # or open the notebooks in VS Code with the .venv kernel
```

The data (`X_2005.txt`, `Y_2005.txt`, ~800 MB) is not in the repository. Run the notebooks in this order:

| Notebook | What it does | Runtime |
|---|---|---|
| `iEMs_Code.ipynb` | Original Random Forest (random row split) | ~5 min |
| `iEMs_UNet.ipynb` | U-Net on hourly maps; train Jan–Aug, validate Sep, test Oct–Dec | ~2 min (GPU) |
| `iEMs_RF_wind.ipynb` | Random Forest with wind speed and wind history, on the same split, plus SHAP | ~12 min |
| `iEMs_SHAP.ipynb` | SHAP feature attribution for the original RF and the U-Net | ~20 min |
| `iEMs_Extremes.ipynb` | Storm days (daily-mean Hm0 above the 90th percentile) and model errors on them | ~2 min |

Run `iEMs_UNet.ipynb` before the other notebooks: it creates the parsed-data cache and the trained weights that they reuse. Plots and tables are saved to `results/` (see [results/README.md](results/README.md)).

## Main results

All models use the same split: train Jan–Aug, validate on Sep, test on Oct–Dec 2005. The test months are never seen in training.

| Model | Inputs | Train RMSE | Val RMSE | **Test RMSE** | Test R² |
|---|---|---|---|---|---|
| Random Forest (original) | Lat, Lon, u10, v10, MSLP | 0.388 | 0.747 | 0.876 | 0.675 |
| Random Forest + wind history | + wind speed; u10, v10, speed, MSLP at 0–24 h before | 0.137 | 0.525 | 0.609 | 0.843 |
| U-Net | same as above, as whole maps | 0.550 | 0.478 | **0.603** | **0.846** |

RMSE in metres.

- **Wind history is what matters most.** Giving the Random Forest the wind from the previous 24 hours cuts its test error by 30%, bringing it level with the U-Net.
- **The original random-row split overstates skill.** It scores the original RF at R² 0.87, but on unseen months the same model gets R² 0.68.
- **Storms are the weak point of both models.** On storm days both under-predict, and the error is about 1.8× larger than on other days.

![Test RMSE and RF importances](results/3_comparison/plots/rf_with_wind_features.png)

![SHAP feature importance](results/4_shap/shap_comparison_all_models.png)

![Model errors on storm days](results/5_extremes/model_errors_storm_vs_other.png)

![Storm days 2005](results/5_extremes/storm_days_timeseries.png)

## Limitations

- Only one year of data (2005).
- Training ran on a single seed.
- The U-Net was early-stopped on a single validation month (September).
