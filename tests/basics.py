import numpy as np


def compare_entries(A, B, sig_figs):
    """
    Compares that every entry in array `A` and `B` match to a certain number
    of *significant figures*.  Note that this is NOT THE SAME AS the number of
    decimal places (from zero); it is actually a more useful check than
    checking decimal places.

    0.5412 == 0.5414       is True if sig_figs = 3, but False if sig_figs = 4
    1.5412 == 1.5414       is True if sig_figs = 4, but False if sig_figs = 5
    1.5412E-5 == 1.5414E-5 is True if sig_figs = 4, but False if sig_figs = 5
    1.5412E+5 == 1.5414E+5 is True if sig_figs = 4, but False if sig_figs = 5

    This function checks that:
    base = np.ceil(np.log10(A[i,j,k]) * np.sign(A[i,j,k]))
    np.abs(A[i,j,k] - B[i,j,k])*1E(base) < 1E(-sig_figs)

    Return
    ------
    Returns a (long) list of boolean comparisons of the entries in `A` and `B`.
    It can then be subsequently checked that np.all(...) entries in this output
    are True, to ensure that the comparison succeeded.
    """
    if not (isinstance(A, np.ndarray)):
        A = np.array([A])
    if not (isinstance(B, np.ndarray)):
        B = np.array([B])
    assert np.prod(A.shape) == np.prod(B.shape)

    check = pow(10, -sig_figs)
    out = []
    for a, b in zip(A.flat, B.flat):
        base = np.ceil(np.log10(a * np.sign(a)))
        out.append(np.abs(a - b) * pow(10, base) < check)

    return out
