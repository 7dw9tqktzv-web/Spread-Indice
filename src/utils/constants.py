"""Enums and constants for the spread trading system."""

from enum import Enum


class Instrument(str, Enum):
    NQ = "NQ"
    ES = "ES"
    RTY = "RTY"
    YM = "YM"


class HedgeMethod(str, Enum):
    OLS_ROLLING = "ols_rolling"
    KALMAN = "kalman"
    DOLLAR_NEUTRAL = "dollar_neutral"
    VOLATILITY_NEUTRAL = "volatility_neutral"


# All 6 pairs as (leg_a, leg_b) tuples
PAIRS = [
    (Instrument.NQ, Instrument.ES),
    (Instrument.NQ, Instrument.RTY),
    (Instrument.NQ, Instrument.YM),
    (Instrument.ES, Instrument.RTY),
    (Instrument.ES, Instrument.YM),
    (Instrument.RTY, Instrument.YM),
]

# Sierra Charts CSV column mapping
SIERRA_COLUMNS = {
    "Date": "date",
    "Time": "time",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Last": "close",
    "Volume": "volume",
    "NumberOfTrades": "num_trades",
    "BidVolume": "bid_vol",
    "AskVolume": "ask_vol",
}

# OHLCV columns after loading
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Raw file naming pattern
RAW_FILE_PATTERN = "{symbol}H26_FUT_CME_1mn.scid_BarData.txt"
