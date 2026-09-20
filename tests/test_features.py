import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"tools"))
import numpy as np
from features import mfcc

def test_shape_and_finite():
    y=mfcc(np.zeros(16000,dtype=np.float32))
    assert y.shape==(49,13)
    assert np.isfinite(y).all()

def test_impulse_changes_features():
    a=np.zeros(16000,dtype=np.float32); b=a.copy(); b[4000]=1
    assert not np.array_equal(mfcc(a),mfcc(b))
