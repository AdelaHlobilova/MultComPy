from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MultComPy as mcp

def test_validate_medium_catches_bad_inputs():
    """Check that validation reliably raises an exception for nonsensical data."""
    
    # 1. Wrong type (regular list instead of numpy array)
    with pytest.raises(TypeError, match="NumPy array"):
        mcp._validate_medium([True, False, True])

    # 2. Wrong number of dimensions (1D array instead of 2D/3D)
    with pytest.raises(ValueError, match="dimensions"):
        mcp._validate_medium(np.array([True, False]))

    # 3. Array contains invalid values (NaN)
    with pytest.raises(ValueError, match="NaN or Inf"):
        bad_array = np.ones((5, 5), dtype=float)
        bad_array[0, 0] = np.nan
        mcp._validate_medium(bad_array)

    # 4. Empty array
    with pytest.raises(ValueError, match="empty"):
        mcp._validate_medium(np.array([]))