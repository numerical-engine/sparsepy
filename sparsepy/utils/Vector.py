import numpy as np
from sparsepy.vector import svector, scvector
import sys
from copy import deepcopy

def to_sparse(d_vec:np.ndarray, dtype:type)->any:
    """Return sparse vector

    Args:
        d_vec (np.ndarray): Vector.
        dtype (type): Type of data
    Returns:
        any: svector if dtype is float, else scvector
    """
    assert len(d_vec.shape) == 1
    indice = np.array([], dtype = int)
    values = np.array([], dtype = dtype)

    for idx, v in enumerate(d_vec):
        if v == 0.:continue
        indice = np.append(indice, idx)
        values = np.append(values, v)

    if dtype == float:
        return svector(dim = len(d_vec), indice = indice, values = values)
    else:
        return scvector(dim = len(d_vec), indice = indice, values = values)


def to_complex(svec:svector)->scvector:
    assert isinstance(svec, svector)
    indice = deepcopy(svec.indice)
    values = deepcopy(svec.values)

    return scvector(dim = svec.dim, indice = indice, values = values.astype(complex))


def dot(svec1:any, svec2:any)->any:
    """Inner product of svcetor

    Args:
        svec1 (svector or scvector): Sparse vector
        svec2 (svector or scvector): Sparse vector
    Returns:
        float or complex: Inner product
    """
    if type(svec1) == type(svec2):
        return svec1@svec2
    else:
        if isinstance(svec1, svector) & isinstance(svec2, scvector):
            svec1 = to_complex(svec1)
        else:
            svec2 = to_complex(svec2)
        return svec1@svec2


def power(svec:any, p:float)->any:
    indice = deepcopy(svec.indice)
    values = deepcopy(svec.values)
    values = np.power(values, p)

    return type(svec)(dim = svec.dim, indice = indice, values = values)


def sum(svec:any)->any:
    return np.sum(svec.values)

def abs(svec:any)->svector:
    indice = deepcopy(svec.indice)
    values = np.abs(svec.values)

    return svector(dim = svec.dim, indice = indice, values = values)

def norm(svec:any, ord:int = 2)->float:
    return np.linalg.norm(svec.values, ord)