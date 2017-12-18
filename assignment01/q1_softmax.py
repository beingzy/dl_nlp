import numpy as np
import pandas as pd


def softmax(x):
    if not isinstance(x):
        x = np.array(x)
    x_exp = np.exp(x)
    return x_exp / sum(x_exp)


def per_row_softmax(mtx):
    """apply softmax to each row of matrix
    """
    is_ndarray = isinstance(mtx, np.ndarray)
    if isinstance(mtx, np.ndarray):
        mtx = pd.DataFrame(mtx)
    elif isinstance(mtx, pd.DataFrame):
        mtx = mtx.copy()
    else:
        raise InputError

    softmax_mtx = mtx.apply(lambda x: softmax(x), axis=1)

    if is_ndarray:
        return softmax_mtx.as_matrix()
    else:
        return softmax_mtx
