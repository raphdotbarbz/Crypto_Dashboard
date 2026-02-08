from pathlib import Path
import yaml
import pandas as pd

def test_settings_yaml():
    cfg = yaml.safe_load(Path("config/settings.yaml").read_text())
    assert "data_path" in cfg
    assert "risk_free" in cfg
    assert "horizons" in cfg

def test_risk_free_loader():
    from src.factors.risk_free import load_risk_free_series
    s = load_risk_free_series("data/risk_free.csv")
    assert not s.empty
    assert abs(s.iloc[0] - (0.0525/365)) < 1e-6

def test_price_store_roundtrip(tmp_path):
    from src.io.loaders import PriceStore
    import pandas as pd
    store = PriceStore(tmp_path)
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC"),
        "price": [100.0, 101.0, 102.0],
    })
    store.put("BTC", df)
    out = store.get("BTC")
    assert len(out) == 3
    assert set(out.columns) == {"date", "price"}
