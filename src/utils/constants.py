"""Enums and constants for the spread trading system."""

from enum import StrEnum


class Instrument(StrEnum):
    # Index futures
    NQ = "NQ"
    ES = "ES"
    RTY = "RTY"
    YM = "YM"
    MNQ = "MNQ"
    MYM = "MYM"
    # Energy futures
    CL = "CL"
    NG = "NG"
    BZ = "BZ"
    HO = "HO"
    RB = "RB"
    MCL = "MCL"
    QG = "QG"
    # Metal futures
    GC = "GC"
    SI = "SI"
    HG = "HG"
    PL = "PL"
    PA = "PA"
    MGC = "MGC"
    SIL = "SIL"
    MHG = "MHG"
    # Grain futures
    ZC = "ZC"
    ZW = "ZW"
    ZS = "ZS"
    MZC = "MZC"
    MZW = "MZW"
    MZS = "MZS"


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
