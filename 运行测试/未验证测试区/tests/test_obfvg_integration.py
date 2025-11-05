import logging
import pandas as pd
from datetime import datetime, timedelta

from 运行测试.未验证测试区.smc_analyzer import SMCDetector
from 运行测试.未验证测试区.config import Config


def _make_dummy_df(rows: int = 40) -> pd.DataFrame:
    now = datetime.utcnow()
    data = []
    price = 100.0
    for i in range(rows):
        # simple synthetic candles with mild trend and volume
        open_p = price
        high_p = price * (1 + 0.003)
        low_p = price * (1 - 0.003)
        close_p = price * (1 + (0.001 if i % 3 != 0 else -0.001))
        volume = 1000 + (i % 10) * 10
        data.append({
            'timestamp': now - timedelta(minutes=(rows - i)),
            'open': open_p,
            'high': high_p,
            'low': low_p,
            'close': close_p,
            'volume': volume,
        })
        price = close_p
    return pd.DataFrame(data)


def test_obfvg_optimized_present_in_detect_smc_structures():
    cfg = Config()
    logger = logging.getLogger("test")
    det = SMCDetector(cfg, logger)
    df = _make_dummy_df()

    res = det.detect_smc_structures(df, '1h')
    assert isinstance(res, dict)
    assert 'ob_fvg_optimized' in res
    overlay = res['ob_fvg_optimized'].get('overlay_result')
    assert isinstance(overlay, dict)
    # required keys
    for k in ['has_overlay', 'overlay_confidence_boost', 'narrow_ob_for_entry', 'wide_ob_for_stop_loss']:
        assert k in overlay


def test_obfvg_optimized_present_in_detect_all_structures():
    cfg = Config()
    logger = logging.getLogger("test")
    det = SMCDetector(cfg, logger)
    df = _make_dummy_df()
    current_price = float(df['close'].iloc[-1])

    res = det.detect_all_structures({'1h': df}, current_price)
    assert isinstance(res, dict)
    assert 'ob_fvg_optimized' in res
    overlay = res['ob_fvg_optimized'].get('overlay_result')
    assert isinstance(overlay, dict)
    for k in ['has_overlay', 'overlay_confidence_boost', 'narrow_ob_for_entry', 'wide_ob_for_stop_loss']:
        assert k in overlay