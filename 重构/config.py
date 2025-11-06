from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Minimal configuration for refactor scaffold.

    Provides only the essentials used across modules to start migrating
    logic into the refactored structure.
    """

    symbol: str = "BTC/USDC:USDC"
    primary_timeframe: str = "3m"
    timeframes: List[str] = field(default_factory=lambda: ["3m", "15m", "1h"])
    data_points: int = 200

    # Caching / data access
    cache_ttl: int = 300

    # Risk basics
    risk_per_trade: float = 0.01
    max_open_positions: int = 5

    # Environment
    simulation_mode: bool = True

    def validate(self) -> None:
        """Basic sanity checks for early failure prevention."""
        if not self.symbol:
            raise ValueError("Config.symbol must be set")
        if self.primary_timeframe not in self.timeframes:
            raise ValueError("primary_timeframe must be in timeframes")