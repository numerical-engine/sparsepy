import numpy as np
from sparsepy.core import sarray
import sys
from copy import deepcopy

def to_sparse(d_vec:np.ndarray)->sarray:
    """numpy配列をsarrayに変換

    Args:
        d_vec (np.ndarray): numpy配列
    Returns:
        sarray: 疎ベクトル
    """
    assert len(d_vec.shape) == 1
    indice = np.array([], dtype = int)
    values = np.array([], dtype = float)

    for idx, v in enumerate(d_vec):
        if v == 0.:continue
        indice = np.append(indice, idx)
        values = np.append(values, v)

    return sarray(dim = len(d_vec), indice = indice, values = values)

def power(svec:sarray, p:float)->sarray:
    indice = deepcopy(svec.indice)
    values = deepcopy(svec.values)
    values = np.power(values, p)

    return sarray(dim = svec.dim, indice = indice, values = values)

def sum(svec:sarray)->float:
    return np.sum(svec.values)

def abs(svec:sarray)->sarray:
    indice = deepcopy(svec.indice)
    values = np.abs(svec.values)

    return sarray(dim = svec.dim, indice = indice, values = values)

def norm(svec:sarray, ord:int = 2)->float:
    return np.linalg.norm(svec.values, ord)