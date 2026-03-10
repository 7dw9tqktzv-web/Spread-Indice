# Commands Reference

All scripts run from project root. Activate venv first: `source venv/Scripts/activate`

## Basics

```bash
pip install -r requirements.txt
python -m pytest tests/ -v --tb=short                    # All tests
python -m pytest tests/test_validation/ -v               # Validation module only
python -m pytest tests/test_hedge/test_ols_rolling.py -v # Single file
```

## Backtest

```bash
python scripts/run_backtest.py --pair NQ_YM --method ols_rolling
python scripts/run_backtest.py --pair NQ_RTY --method kalman --alpha-ratio 3e-7
```

## Utilities

```bash
python scripts/validate_numba.py                         # Numba vs Python parity
python scripts/plot_equity_curves.py                     # Equity curve visualization
python scripts/mfe_mae_analysis.py                       # MFE/MAE trade analysis
```
