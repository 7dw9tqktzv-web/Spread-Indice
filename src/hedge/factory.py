"""Factory for hedge ratio estimators."""

from src.hedge.base import HedgeRatioEstimator
from src.hedge.kalman import KalmanConfig, KalmanEstimator
from src.hedge.ols_rolling import OLSRollingConfig, OLSRollingEstimator
from src.utils.constants import HedgeMethod

_REGISTRY: dict[HedgeMethod, type[HedgeRatioEstimator]] = {
    HedgeMethod.OLS_ROLLING: OLSRollingEstimator,
    HedgeMethod.KALMAN: KalmanEstimator,
}

_CONFIG_REGISTRY: dict[HedgeMethod, type] = {
    HedgeMethod.OLS_ROLLING: OLSRollingConfig,
    HedgeMethod.KALMAN: KalmanConfig,
}


def create_estimator(method: str | HedgeMethod, **kwargs) -> HedgeRatioEstimator:
    """Create a hedge ratio estimator by method name.

    Parameters
    ----------
    method : str or HedgeMethod
        Method name (e.g. "ols_rolling", "kalman").
    **kwargs
        Passed to the config dataclass constructor.

    Examples
    --------
    >>> estimator = create_estimator("ols_rolling", window=7200, zscore_window=12)
    >>> estimator = create_estimator(HedgeMethod.KALMAN, alpha_ratio=1e-5)
    """
    key = HedgeMethod(method) if isinstance(method, str) else method

    if key not in _REGISTRY:
        raise ValueError(f"Unknown hedge method: {method!r}. Available: {list(_REGISTRY)}")

    config_cls = _CONFIG_REGISTRY[key]
    config = config_cls(**kwargs) if kwargs else config_cls()
    return _REGISTRY[key](config=config)
