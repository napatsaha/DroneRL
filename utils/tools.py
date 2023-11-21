def clamp(n, minn, maxn):
    """Clamp scalar value between maximum and minimum."""
    return max(min(maxn, n), minn)


def safe_simplify(item):
    """
    Return only first element in a single-item list or array.
    Otherwise return the original item.
    """
    if '__len__' in item.__dir__():
        if len(item) == 1:
            return item[0]
    return item


def identity(a, b):
    """
    Returns the first of the two arguments no matter what.
    Useful as a counterpart to min() and max().
    """
    return a