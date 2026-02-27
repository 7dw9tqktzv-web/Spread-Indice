"""Enums and constants for the spread trading system."""

from enum import StrEnum


class Instrument(StrEnum):
    NQ = "NQ"
    ES = "ES"
    RTY = "RTY"
    YM = "YM"
    MNQ = "MNQ"
    MYM = "MYM"


class HedgeMethod(StrEnum):
    OLS_ROLLING = "ols_rolling"
    KALMAN = "kalman"


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

# Raw file naming pattern
RAW_FILE_PATTERN = "{symbol}H26_FUT_CME_1mn.scid_BarData.txt"
