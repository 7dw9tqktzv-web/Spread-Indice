# Commands Reference

All scripts run from project root. Activate venv first: `source venv/Scripts/activate`

## Basics

```bash
pip install -r requirements.txt
python -m pytest tests/ -v --tb=short                    # All 62 tests
python -m pytest tests/ -v --ignore=tests/test_integration  # Unit only
python -m pytest tests/test_hedge/test_ols_rolling.py -v    # Single file
```

## Backtest

```bash
python scripts/run_backtest.py --pair NQ_YM --method ols_rolling
python scripts/run_backtest.py --pair NQ_RTY --method kalman --alpha-ratio 3e-7
```

## Grid Search -- NQ_YM

```bash
python scripts/run_refined_grid.py --workers 5          # OLS refined (1,080,000 combos)
python scripts/run_grid_kalman_v3.py --workers 10       # Kalman (1,009,800 combos)
python scripts/run_grid.py --workers 20                 # OLS broad (43,200, 6 pairs)
```

## Grid Search -- NQ_RTY

```bash
python scripts/refine_ols_balanced.py --workers 10      # OLS refined (14,374,656 combos)
python scripts/run_grid_kalman_NQ_RTY.py --workers 10   # Kalman (856,800 combos)
python scripts/run_grid_ols_NQ_RTY.py --workers 10      # OLS broad (15,945,930 combos)
```

## Validation

```bash
python scripts/validate_top5_configs.py                 # OLS NQ_YM
python scripts/validate_kalman_top.py                   # Kalman NQ_YM (IS/OOS + WF)
python scripts/validate_NQ_RTY_top6.py                  # OLS NQ_RTY (6 configs)
python scripts/validate_top_NQ_RTY.py                   # Kalman NQ_RTY
python scripts/validate_confidence.py                   # Quant validation originale
python scripts/validate_numba.py                        # Numba vs Python parity
```

## Analysis

```bash
python scripts/analyze_grid_results.py                  # Deep analysis NQ_YM
python scripts/analyze_grid_NQ_RTY.py                   # Kalman NQ_RTY dimensionnel
python scripts/rank_top_configs_NQ_RTY.py               # Multi-criteria ranking NQ_RTY
python scripts/compare_candidates_NQ_RTY.py             # Overlap analysis NQ_RTY
python scripts/check_maxdd_refined_NQ_RTY.py            # MaxDD check NQ_RTY
python scripts/find_safe_kalman.py                      # E-mini safe configs
python scripts/find_safe_kalman_micro.py                # Micro contracts (x1/x2/x3)
python scripts/analyze_2023_losses.py                   # Autopsie 2023
python scripts/analyze_filters.py                       # Filter ablation NQ_YM
```

## Diagnostics

```bash
python scripts/test_adaptive_r.py                       # P1 R adaptatif (INVALIDE)
python scripts/analyze_kalman_diagnostics.py            # P2 diagnostics + deconfounding
```

## Phase 2a Sierra Validation

```bash
python scripts/validate_sierra_v3.py                    # Bar-a-bar metriques brutes (Pearson r)
python scripts/validate_signal_parity.py                # Parite signaux (regime + state machine)
python scripts/debug_hurst_final.py                     # Debug Hurst C++
```
