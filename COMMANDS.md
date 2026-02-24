# Commands Reference

All scripts run from project root. Activate venv first: `source venv/Scripts/activate`

## Basics

```bash
pip install -r requirements.txt
python -m pytest tests/ -v --tb=short                    # All 107 tests
python -m pytest tests/test_validation/ -v               # Validation module only
python -m pytest tests/test_hedge/test_ols_rolling.py -v # Single file
```

## Backtest

```bash
python scripts/run_backtest.py --pair NQ_YM --method ols_rolling
python scripts/run_backtest.py --pair NQ_RTY --method kalman --alpha-ratio 3e-7
```

## Phase 13c -- Config D NQ_YM (Binary Gates + CPCV)

```bash
# Grid massif (24.7M combos, ~3h50m with 20 workers)
python scripts/phase13c_grid_massif.py --dry-run         # Preview grid dimensions
python scripts/phase13c_grid_massif.py --workers 20      # Run full grid

# Analysis pipeline
python scripts/phase13c_analysis.py                      # Dimensional analysis + top N
python scripts/phase13c_deep_analysis.py                 # Overlay + autopsy + WF recency
python scripts/phase13c_surgical_grid.py                 # Surgical grid (time_stop x delta_sl)
python scripts/phase13c_final_checks.py                  # Monte Carlo + slippage sensitivity
python scripts/phase13c_report_d.py                      # HTML report -> output/NQ_YM/config_D_reference.html
```

## Grid Search -- NQ_RTY

```bash
python scripts/refine_ols_balanced.py --workers 10       # OLS refined (14.4M combos)
python scripts/run_grid_ols_NQ_RTY.py --workers 10       # OLS broad (15.9M combos)
python scripts/run_grid_kalman_NQ_RTY.py --workers 10    # Kalman v1 (856K combos)
python scripts/grid_kalman_v2_NQ_RTY.py --workers 10     # Kalman v2 (290K combos, corrected weights)
```

## Validation -- NQ_RTY

```bash
python scripts/validate_NQ_RTY_top6.py                   # OLS top 6 (IS/OOS + WF)
python scripts/validate_top_NQ_RTY.py                    # Kalman NQ_RTY
python scripts/validate_kalman_v2_NQ_RTY.py              # Kalman v2 (7 configs)
python scripts/step3_maxdd_overlap_NQ_RTY.py             # MaxDD + overlap analysis
python scripts/step4_timestop_hourly_NQ_RTY.py           # Time stop + hourly deep-dive
python scripts/step5_validate_NQ_RTY.py                  # Full validation (IS/OOS + WF + Perm)
python scripts/step6_autopsy_NQ_RTY.py                   # Trade autopsy
python scripts/step7_micro_propfirm_NQ_RTY.py            # Micro contracts + propfirm
```

## Analysis -- NQ_RTY

```bash
python scripts/analyze_grid_NQ_RTY.py                    # Kalman dimensionnel
python scripts/analyze_grid_ols_NQ_RTY.py                # OLS dimensionnel
python scripts/analyze_grid_step2_NQ_RTY.py              # Grid step 2 (48K candidates)
python scripts/rank_top_configs_NQ_RTY.py                # Multi-criteria ranking
python scripts/compare_candidates_NQ_RTY.py              # Overlap analysis
python scripts/check_maxdd_refined_NQ_RTY.py             # MaxDD check
python scripts/analyze_kalman_textbox_NQ_RTY.py          # Kalman textbox dashboard
python scripts/ablation_conf_weights_NQ_RTY.py           # Confidence weight ablation
```

## Utilities

```bash
python scripts/validate_numba.py                         # Numba vs Python parity
python scripts/plot_equity_curves.py                     # Equity curve visualization
```

## Phase 2 Sierra Validation

```bash
python scripts/validate_sierra_v3.py                     # Bar-a-bar metriques brutes (Pearson r)
python scripts/validate_signal_parity.py                 # Parite signaux (regime + state machine)
python scripts/debug_hurst_final.py                      # Debug Hurst C++
```

## Archived Scripts

18 superseded scripts in `scripts/archive/` (Phases 6-13a). Reference only, not for active use.
