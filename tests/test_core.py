import pytest
import pandas as pd
import numpy as np
from covered_calls.features.resistance import detect_resistance
from covered_calls.options.cleaning import clean_options

def test_cleaning():
    data = {
        'bid': [1.0, 1.2, 0.0],
        'ask': [1.1, 1.0, 0.5], # 2nd is crossed, 3rd zero bid
        'mid': [1.05, 1.1, 0.25],
        'openInterest': [100, 100, 100],
        'strike': [100, 105, 110]
    }
    df = pd.DataFrame(data)
    cleaned = clean_options(df, min_oi=10)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['mid'] == 1.05

def test_resistance():
    # Mock Data
    daily = pd.DataFrame({
        'high': [100, 105, 110, 108, 102],
        'close': [98, 104, 109, 107, 101]
    })
    intraday = pd.DataFrame()
    current = 101
    
    res = detect_resistance(daily, intraday, current)
    # 5-day max is 110
    # Should detect 110
    assert not res.empty
    assert res.iloc[0]['level'] >= 101