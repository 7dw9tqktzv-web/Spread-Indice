"""Chicago Time helpers and session boundary utilities."""

from dataclasses import dataclass
from datetime import time

import pandas as pd


@dataclass(frozen=True)
class SessionConfig:
    """Defines trading session boundaries in Chicago Time."""
    session_start: time = time(17, 30)   # Globex open
    session_end: time = time(15, 30)     # Globex close
    buffer_minutes: int = 30             # Exclude first/last N minutes
    trading_start: time = time(4, 0)     # Trading window start
    trading_end: time = time(14, 0)      # Trading window end


def parse_session_config(cfg: dict) -> SessionConfig:
    """Parse session config from YAML dict."""
    def _parse_time(s: str) -> time:
        parts = s.split(":")
        return time(int(parts[0]), int(parts[1]))

    return SessionConfig(
        session_start=_parse_time(cfg["session_start"]),
        session_end=_parse_time(cfg["session_end"]),
        buffer_minutes=cfg["buffer_minutes"],
        trading_start=_parse_time(cfg["trading_start"]),
        trading_end=_parse_time(cfg["trading_end"]),
    )


def is_in_trading_window(ts: pd.Timestamp, session: SessionConfig) -> bool:
    """Check if a timestamp falls within the trading window."""
    t = ts.time()
    return session.trading_start <= t < session.trading_end


def filter_trading_window(df: pd.DataFrame, session: SessionConfig) -> pd.DataFrame:
    """Keep only bars within the trading window [trading_start, trading_end)."""
    mask = df.index.map(lambda ts: session.trading_start <= ts.time() < session.trading_end)
    return df.loc[mask].copy()


def filter_session(df: pd.DataFrame, session: SessionConfig) -> pd.DataFrame:
    """Keep only bars within the Globex session, excluding buffer periods.

    Session spans overnight: session_start (17:30) -> session_end (15:30).
    Buffer excludes first and last buffer_minutes of the session.
    """
    buffer = pd.Timedelta(minutes=session.buffer_minutes)

    # Buffered session boundaries
    buf_start = (
        pd.Timestamp("2000-01-01") + pd.Timedelta(hours=session.session_start.hour,
                                                    minutes=session.session_start.minute)
        + buffer
    ).time()
    buf_end = (
        pd.Timestamp("2000-01-01") + pd.Timedelta(hours=session.session_end.hour,
                                                    minutes=session.session_end.minute)
        - buffer
    ).time()

    def _in_session(t: time) -> bool:
        # Overnight session: after buf_start OR before buf_end
        return t >= buf_start or t < buf_end

    mask = df.index.map(lambda ts: _in_session(ts.time()))
    return df.loc[mask].copy()
