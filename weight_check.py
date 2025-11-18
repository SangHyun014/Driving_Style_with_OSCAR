#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick check: verify if any weights in a Keras .h5 file are NaN or Inf
"""

import h5py
import numpy as np
import sys

def check_h5_weights(path):
    bad = []
    with h5py.File(path, 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                try:
                    arr = obj[()]
                    if hasattr(arr, 'dtype') and np.issubdtype(arr.dtype, np.number):
                        if not np.isfinite(arr).all():
                            bad.append((name,
                                        float(np.nanmin(arr)) if np.isnan(arr).any() else np.min(arr),
                                        float(np.nanmax(arr)) if np.isnan(arr).any() else np.max(arr)))
                except Exception as e:
                    bad.append((name, "read_error", str(e)))
        f.visititems(visit)

    if bad:
        print("[WARN] {} has NaN/Inf in the following datasets:".format(path))
        for b in bad:
            print("  {:<60s}  min={}  max={}".format(b[0], b[1], b[2]))
    else:
        print("[OK]  {} weights are all finite.".format(path))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_weights_nan.py <path_to_weights.h5>")
        sys.exit(1)
    check_h5_weights(sys.argv[1])
