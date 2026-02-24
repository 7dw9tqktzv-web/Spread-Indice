"""Deflated Sharpe Ratio for multiple testing correction.

Corrects for the number of configurations tested (selection bias).
Reference: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".

DSR = prob( SR* > 0 | SR_0 = E[max(SR_k)] )
where SR_0 is the expected maximum Sharpe under the null hypothesis.
"""

import numpy as np
from scipy import stats


def expected_max_sharpe(
    n_trials: int,
    std_sharpe: float = 1.0,
    mean_sharpe: float = 0.0,
) -> float:
    """Expected maximum Sharpe ratio under H0 (no true edge).

    Uses Gumbel approximation:
    E[max] ~ mean + std * (sqrt(2*log(N)) - (log(pi)+log(log(N))) / (2*sqrt(2*log(N))))

    Parameters
    ----------
    n_trials : int
        Number of configurations tested.
    std_sharpe : float
        Standard deviation of Sharpe ratios across trials.
    mean_sharpe : float
        Mean of Sharpe ratios across trials.
    """
    if n_trials <= 1:
        return mean_sharpe
    log_n = np.log(n_trials)
    sqrt_2logn = np.sqrt(2 * log_n)
    e_max = sqrt_2logn - (np.log(np.pi) + np.log(log_n)) / (2 * sqrt_2logn)
    return mean_sharpe + std_sharpe * e_max


def deflated_sharpe_ratio(
    observed_sharpe: float,
    sr_benchmark: float,
    n_trades: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute the Deflated Sharpe Ratio (probability that true SR > benchmark).

    Uses the distribution of SR estimator which accounts for non-normality.

    Parameters
    ----------
    observed_sharpe : float
        The observed (sample) Sharpe ratio.
    sr_benchmark : float
        The benchmark SR0 = E[max(SR)] from multiple testing.
    n_trades : int
        Number of trades (observations).
    skewness : float
        Skewness of returns/PnLs.
    kurtosis : float
        Raw kurtosis of returns/PnLs (normal = 3.0).

    Returns
    -------
    float : probability (0 to 1). DSR > 0.95 is conventionally significant.
    """
    if n_trades < 3:
        return 0.0

    sr = observed_sharpe
    excess_kurt = kurtosis - 3.0
    var_sr = (1.0 - skewness * sr + (excess_kurt / 4.0) * sr**2) / (n_trades - 1)

    if var_sr <= 0:
        return 0.0

    se_sr = np.sqrt(var_sr)
    z = (sr - sr_benchmark) / se_sr
    return float(stats.norm.cdf(z))


def compute_dsr_for_config(
    observed_sharpe: float,
    trade_pnls: np.ndarray,
    n_trials: int,
    all_sharpes: np.ndarray,
) -> dict:
    """Full DSR computation for a single selected configuration.

    Parameters
    ----------
    observed_sharpe : float
        Sharpe of the selected config (typically CPCV median Sharpe).
    trade_pnls : np.ndarray
        PnL array of the selected config's trades.
    n_trials : int
        Total configs tested in the grid.
    all_sharpes : np.ndarray
        Array of median CPCV Sharpes for all configs tested.
    """
    n_trades = len(trade_pnls)
    if n_trades < 3:
        return {
            "dsr": 0.0,
            "sr_benchmark": 0.0,
            "observed_sharpe": 0.0,
            "n_trades": 0,
            "skewness": 0.0,
            "kurtosis": 3.0,
        }

    skew = float(stats.skew(trade_pnls))
    kurt = float(stats.kurtosis(trade_pnls, fisher=False))  # raw kurtosis

    valid_sharpes = all_sharpes[np.isfinite(all_sharpes)]
    mean_sr = float(valid_sharpes.mean()) if len(valid_sharpes) > 0 else 0.0
    std_sr = float(valid_sharpes.std()) if len(valid_sharpes) > 1 else 1.0

    sr0 = expected_max_sharpe(n_trials, std_sr, mean_sr)
    dsr = deflated_sharpe_ratio(observed_sharpe, sr0, n_trades, skew, kurt)

    return {
        "dsr": round(dsr, 4),
        "sr_benchmark": round(sr0, 4),
        "observed_sharpe": round(observed_sharpe, 4),
        "n_trades": n_trades,
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
    }
