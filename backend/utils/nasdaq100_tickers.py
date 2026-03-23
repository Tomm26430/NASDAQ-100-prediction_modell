"""
NASDAQ-100 constituent tickers plus the index symbol used on Yahoo Finance (^NDX).

The constituent list matches Wikipedia's "Current components" table (as of Jan 2026).
The index has 100 companies; some firms have two share classes (e.g. GOOGL and GOOG),
which is why you see more than 100 lines in a simple company count.

For a live-updated list, see: https://www.nasdaq.com/market-activity/quotes/nasdaq-ndx-index
"""

from config import settings

# All NASDAQ-100 symbols we track (share classes included). Sorted A–Z for stable APIs/UI.
NASDAQ100_TICKERS: list[str] = sorted(
    [
        "ADBE",
        "ADI",
        "ADP",
        "ADSK",
        "AEP",
        "AMAT",
        "AMD",
        "AMGN",
        "AMZN",
        "APP",
        "ARM",
        "ASML",
        "AVGO",
        "AXON",
        "ABNB",
        "AAPL",
        "ALNY",
        "BKR",
        "BKNG",
        "CDNS",
        "CCEP",
        "CEG",
        "CHTR",
        "CMCSA",
        "COST",
        "CPRT",
        "CRWD",
        "CSGP",
        "CSX",
        "CTAS",
        "CTSH",
        "CSCO",
        "DASH",
        "DDOG",
        "DXCM",
        "EA",
        "EXC",
        "FAST",
        "FANG",
        "FER",
        "FTNT",
        "GEHC",
        "GILD",
        "GOOG",
        "GOOGL",
        "HON",
        "IDXX",
        "INSM",
        "INTC",
        "INTU",
        "ISRG",
        "KDP",
        "KHC",
        "KLAC",
        "LIN",
        "LRCX",
        "MAR",
        "MCHP",
        "MDLZ",
        "MELI",
        "META",
        "MNST",
        "MPWR",
        "MSFT",
        "MU",
        "MRVL",
        "MSTR",
        "NFLX",
        "NVDA",
        "NXPI",
        "ODFL",
        "ORLY",
        "PANW",
        "PAYX",
        "PCAR",
        "PDD",
        "PEP",
        "PLTR",
        "PYPL",
        "QCOM",
        "REGN",
        "ROP",
        "ROST",
        "SBUX",
        "SHOP",
        "SNPS",
        "STX",
        "TEAM",
        "TMUS",
        "TRI",
        "TTWO",
        "TSLA",
        "TXN",
        "VRSK",
        "VRTX",
        "WBD",
        "WDAY",
        "WDC",
        "WMT",
        "XEL",
        "ZS",
    ]
)

# Yahoo Finance symbol for the Nasdaq-100 price index (not a tradable stock).
NDX_INDEX_SYMBOL = "^NDX"

# Small universe for quick end-to-end tests (matches the project brief).
LIGHT_MODE_TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", NDX_INDEX_SYMBOL]


def all_tracked_symbols() -> list[str]:
    """Return every ticker we support: all NASDAQ-100 names plus the ^NDX index."""
    # Index last so dashboards can show "stocks" then the index, if you iterate in order.
    return list(NASDAQ100_TICKERS) + [NDX_INDEX_SYMBOL]


def get_active_tickers() -> list[str]:
    """
    Symbols to fetch, cache, and (later) train on.

    Controlled by LIGHT_MODE in config: when True, only five liquid names + the index.
    """
    if settings.LIGHT_MODE:
        return list(LIGHT_MODE_TICKERS)
    return all_tracked_symbols()
