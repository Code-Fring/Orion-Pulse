"""Company name to ticker symbol resolver for Orion Pulse."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolInfo:
    """Information about a symbol/company."""

    symbol: str
    company_name: str
    sector: str | None = None
    industry: str | None = None
    exchange: str | None = None
    aliases: list[str] = None

    def __post_init__(self) -> None:
        if self.aliases is None:
            object.__setattr__(self, "aliases", [])

    def matches(self, query: str) -> bool:
        """Check if query matches this symbol."""
        query_lower = query.lower().strip()
        # Exact symbol match
        if query_lower == self.symbol.lower():
            return True
        # Company name match
        if query_lower == self.company_name.lower():
            return True
        # Alias match
        for alias in self.aliases:
            if query_lower == alias.lower():
                return True
        # Partial match on company name
        if query_lower in self.company_name.lower():
            return True
        # Partial match on symbol
        return query_lower in self.symbol.lower()

    def display_name(self) -> str:
        """Get display name with both company and symbol."""
        return f"{self.company_name} ({self.symbol})"


class SymbolResolver:
    """Resolve company names to ticker symbols."""

    def __init__(self) -> None:
        self._symbols: dict[str, SymbolInfo] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default symbol mappings."""
        # Major US tech stocks
        symbols = [
            SymbolInfo(
                symbol="NVDA",
                company_name="NVIDIA Corporation",
                sector="Technology",
                industry="Semiconductors",
                exchange="NASDAQ",
                aliases=["nvidia", "nvidia corp"],
            ),
            SymbolInfo(
                symbol="AAPL",
                company_name="Apple Inc.",
                sector="Technology",
                industry="Consumer Electronics",
                exchange="NASDAQ",
                aliases=["apple", "apple inc"],
            ),
            SymbolInfo(
                symbol="MSFT",
                company_name="Microsoft Corporation",
                sector="Technology",
                industry="Software",
                exchange="NASDAQ",
                aliases=["microsoft", "msft"],
            ),
            SymbolInfo(
                symbol="GOOGL",
                company_name="Alphabet Inc. Class A",
                sector="Technology",
                industry="Internet Services",
                exchange="NASDAQ",
                aliases=["google", "alphabet", "googl", "goog"],
            ),
            SymbolInfo(
                symbol="GOOG",
                company_name="Alphabet Inc. Class C",
                sector="Technology",
                industry="Internet Services",
                exchange="NASDAQ",
                aliases=["alphabet class c"],
            ),
            SymbolInfo(
                symbol="AMZN",
                company_name="Amazon.com Inc.",
                sector="Consumer Cyclical",
                industry="E-Commerce",
                exchange="NASDAQ",
                aliases=["amazon", "amzn"],
            ),
            SymbolInfo(
                symbol="META",
                company_name="Meta Platforms Inc.",
                sector="Technology",
                industry="Social Media",
                exchange="NASDAQ",
                aliases=["meta", "facebook", "fb"],
            ),
            SymbolInfo(
                symbol="TSLA",
                company_name="Tesla Inc.",
                sector="Consumer Cyclical",
                industry="Auto Manufacturers",
                exchange="NASDAQ",
                aliases=["tesla", "tsla"],
            ),
            SymbolInfo(
                symbol="NFLX",
                company_name="Netflix Inc.",
                sector="Communication Services",
                industry="Entertainment",
                exchange="NASDAQ",
                aliases=["netflix", "nflx"],
            ),
            SymbolInfo(
                symbol="AMD",
                company_name="Advanced Micro Devices Inc.",
                sector="Technology",
                industry="Semiconductors",
                exchange="NASDAQ",
                aliases=["amd", "advanced micro devices"],
            ),
            SymbolInfo(
                symbol="INTC",
                company_name="Intel Corporation",
                sector="Technology",
                industry="Semiconductors",
                exchange="NASDAQ",
                aliases=["intel", "intc"],
            ),
            SymbolInfo(
                symbol="CRM",
                company_name="Salesforce Inc.",
                sector="Technology",
                industry="Software",
                exchange="NYSE",
                aliases=["salesforce", "crm"],
            ),
            SymbolInfo(
                symbol="ORCL",
                company_name="Oracle Corporation",
                sector="Technology",
                industry="Software",
                exchange="NYSE",
                aliases=["oracle", "orcl"],
            ),
            SymbolInfo(
                symbol="ADBE",
                company_name="Adobe Inc.",
                sector="Technology",
                industry="Software",
                exchange="NASDAQ",
                aliases=["adobe", "adbe"],
            ),
            SymbolInfo(
                symbol="PYPL",
                company_name="PayPal Holdings Inc.",
                sector="Financial Services",
                industry="Credit Services",
                exchange="NASDAQ",
                aliases=["paypal", "pypl"],
            ),
            SymbolInfo(
                symbol="UBER",
                company_name="Uber Technologies Inc.",
                sector="Technology",
                industry="Software",
                exchange="NYSE",
                aliases=["uber", "ubr"],
            ),
            SymbolInfo(
                symbol="SHOP",
                company_name="Shopify Inc.",
                sector="Technology",
                industry="Software",
                exchange="NYSE",
                aliases=["shopify", "shop"],
            ),
            SymbolInfo(
                symbol="SQ",
                company_name="Block Inc.",
                sector="Technology",
                industry="Software",
                exchange="NYSE",
                aliases=["block", "square", "sq"],
            ),
            SymbolInfo(
                symbol="ZM",
                company_name="Zoom Video Communications Inc.",
                sector="Technology",
                industry="Software",
                exchange="NASDAQ",
                aliases=["zoom", "zm"],
            ),
            SymbolInfo(
                symbol="SNOW",
                company_name="Snowflake Inc.",
                sector="Technology",
                industry="Software",
                exchange="NYSE",
                aliases=["snowflake", "snow"],
            ),
            SymbolInfo(
                symbol="PLTR",
                company_name="Palantir Technologies Inc.",
                sector="Technology",
                industry="Software",
                exchange="NYSE",
                aliases=["palantir", "pltr"],
            ),
            SymbolInfo(
                symbol="COIN",
                company_name="Coinbase Global Inc.",
                sector="Financial Services",
                industry="Capital Markets",
                exchange="NASDAQ",
                aliases=["coinbase", "coin"],
            ),
            SymbolInfo(
                symbol="MSTR",
                company_name="MicroStrategy Inc.",
                sector="Technology",
                industry="Software",
                exchange="NASDAQ",
                aliases=["microstrategy", "mstr"],
            ),
            SymbolInfo(
                symbol="RIOT",
                company_name="Riot Platforms Inc.",
                sector="Technology",
                industry="Software",
                exchange="NASDAQ",
                aliases=["riot", "riot blockchain"],
            ),
            SymbolInfo(
                symbol="MARA",
                company_name="Marathon Digital Holdings Inc.",
                sector="Technology",
                industry="Software",
                exchange="NASDAQ",
                aliases=["marathon", "mara"],
            ),
            # Major indices
            SymbolInfo(
                symbol="SPY",
                company_name="SPDR S&P 500 ETF Trust",
                sector="ETF",
                industry="Broad Market",
                exchange="NYSE Arca",
                aliases=["sp500", "s&p 500", "spy"],
            ),
            SymbolInfo(
                symbol="QQQ",
                company_name="Invesco QQQ Trust",
                sector="ETF",
                industry="Technology",
                exchange="NASDAQ",
                aliases=["nasdaq 100", "qqq"],
            ),
            SymbolInfo(
                symbol="DIA",
                company_name="SPDR Dow Jones Industrial Average ETF",
                sector="ETF",
                industry="Broad Market",
                exchange="NYSE Arca",
                aliases=["dow jones", "dow", "dia"],
            ),
            SymbolInfo(
                symbol="IWM",
                company_name="iShares Russell 2000 ETF",
                sector="ETF",
                industry="Small Cap",
                exchange="NYSE Arca",
                aliases=["russell 2000", "small cap", "iwm"],
            ),
            # Crypto-related (via stocks)
            SymbolInfo(
                symbol="BTC-USD",
                company_name="Bitcoin USD",
                sector="Cryptocurrency",
                industry="Digital Asset",
                exchange="Crypto",
                aliases=["bitcoin", "btc"],
            ),
            SymbolInfo(
                symbol="ETH-USD",
                company_name="Ethereum USD",
                sector="Cryptocurrency",
                industry="Digital Asset",
                exchange="Crypto",
                aliases=["ethereum", "eth"],
            ),
        ]

        for sym in symbols:
            self.register(sym)

    def register(self, symbol_info: SymbolInfo) -> None:
        """Register a symbol."""
        key = symbol_info.symbol.upper()
        self._symbols[key] = symbol_info

    def resolve(self, query: str) -> SymbolInfo | None:
        """Resolve a query to a SymbolInfo."""
        query_clean = query.strip().upper()

        # Direct symbol lookup
        if query_clean in self._symbols:
            return self._symbols[query_clean]

        # Search by match
        matches = []
        for sym in self._symbols.values():
            if sym.matches(query):
                matches.append(sym)

        if not matches:
            return None

        # Prefer exact symbol match
        for m in matches:
            if m.symbol.upper() == query_clean:
                return m

        # Prefer exact company name match
        for m in matches:
            if m.company_name.lower() == query.lower().strip():
                return m

        # Return first match
        return matches[0]

    def search(self, query: str, limit: int = 10) -> list[SymbolInfo]:
        """Search for symbols matching query."""
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        results = []
        for sym in self._symbols.values():
            # Score the match
            score = 0
            if sym.symbol.lower().startswith(query_lower):
                score = 100
            elif sym.company_name.lower().startswith(query_lower):
                score = 90
            elif query_lower in sym.symbol.lower():
                score = 50
            elif query_lower in sym.company_name.lower():
                score = 40
            else:
                for alias in sym.aliases:
                    if query_lower in alias.lower():
                        score = 30
                        break

            if score > 0:
                results.append((score, sym))

        # Sort by score descending
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:limit]]

    def get(self, symbol: str) -> SymbolInfo | None:
        """Get symbol info by exact symbol."""
        return self._symbols.get(symbol.upper())

    def all(self) -> list[SymbolInfo]:
        """Get all registered symbols."""
        return list(self._symbols.values())

    def get_by_sector(self, sector: str) -> list[SymbolInfo]:
        """Get symbols by sector."""
        return [
            s
            for s in self._symbols.values()
            if s.sector and s.sector.lower() == sector.lower()
        ]


# Global resolver instance
_resolver: SymbolResolver | None = None


def get_resolver() -> SymbolResolver:
    """Get global symbol resolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = SymbolResolver()
    return _resolver


def resolve_symbol(query: str) -> SymbolInfo | None:
    """Convenience function to resolve a symbol."""
    return get_resolver().resolve(query)


def search_symbols(query: str, limit: int = 10) -> list[SymbolInfo]:
    """Convenience function to search symbols."""
    return get_resolver().search(query, limit)


def format_symbol_info(info: SymbolInfo, no_color: bool = False) -> str:
    """Format symbol info for display."""
    from rich.text import Text

    text = Text()
    text.append(info.company_name, style="bold white")
    text.append("  ")
    text.append(info.symbol, style="bold cyan")
    if info.sector:
        text.append("  ")
        text.append(f"{info.sector}", style="dim")
        if info.industry:
            text.append(f" · {info.industry}", style="dim")
    return text
