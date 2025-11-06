from 重构.position_manager import PositionManager


def test_create_position_and_retrieve():
    pm = PositionManager()
    pos = pm.create_position("BTC/USDC:USDC", 100.0, 0.01)
    assert pos["status"] == "OPEN"
    assert pos["id"].startswith("pos-")
    assert pm.get_positions(open_only=True)[0]["symbol"] == "BTC/USDC:USDC"


def test_update_position_hits_stop_loss():
    pm = PositionManager()
    pos = pm.create_position("BTC/USDC:USDC", 100.0, 0.01, stop_loss=95.0)
    updated = pm.update_position(pos["id"], 94.0)
    assert updated["status"] == "CLOSED"
    assert updated["closed_at"] is not None


def test_update_position_hits_take_profit():
    pm = PositionManager()
    pos = pm.create_position("BTC/USDC:USDC", 100.0, 0.01, take_profit=105.0)
    updated = pm.update_position(pos["id"], 106.0)
    assert updated["status"] == "CLOSED"
    assert updated["closed_at"] is not None

