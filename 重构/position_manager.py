from typing import Dict, Any, List, Optional
from datetime import datetime


class PositionManager:
    """Minimal position lifecycle manager for refactor scaffold."""

    def __init__(self) -> None:
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._seq: int = 0

    def create_position(self, symbol: str, entry_price: float, amount: float,
                        stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict[str, Any]:
        self._seq += 1
        pid = f"pos-{self._seq}"
        pos = {
            "id": pid, "symbol": symbol,
            "entry_price": float(entry_price), "amount": float(amount),
            "stop_loss": stop_loss, "take_profit": take_profit,
            "status": "OPEN", "opened_at": datetime.utcnow().isoformat(),
            "closed_at": None, "last_price": entry_price,
        }
        self._positions[pid] = pos
        return pos

    def update_position(self, pid: str, current_price: float) -> Dict[str, Any]:
        pos = self._positions.get(pid)
        if not pos:
            raise KeyError(f"Position {pid} not found")
        pos["last_price"] = float(current_price)
        # Minimal exit logic
        if pos.get("take_profit") and current_price >= pos["take_profit"]:
            pos["status"] = "CLOSED"; pos["closed_at"] = datetime.utcnow().isoformat()
        elif pos.get("stop_loss") and current_price <= pos["stop_loss"]:
            pos["status"] = "CLOSED"; pos["closed_at"] = datetime.utcnow().isoformat()
        return pos

    def get_positions(self, open_only: bool = False) -> List[Dict[str, Any]]:
        vals = list(self._positions.values())
        return [p for p in vals if p["status"] == "OPEN"] if open_only else vals