"""
This demo has been split into three per-algorithm demos:

    examples/demo_alg1_modified_cholesky.py     OLS via augmented Gram + LU
    examples/demo_alg2_sgso.py                  Simplified Gram-Schmidt
    examples/demo_alg3_weighted_gi.py           Weighted generalised inverse

Each one simulates a linear system with noise, runs the algorithm, and
compares to a textbook reference (Householder QR for OLS, Gram-Schmidt for
orthogonalisation, generalised least squares for weighted regression).
"""
print(__doc__)
