import numpy as np

from sparsepy.utils import Array
from sparsepy.core import sarray, smatrix
from sparsepy.matrix.COO import coo
from sparsepy.utils import COO_utils


def Ax(A:smatrix, x:sarray)->sarray:
    """Axの値を出力

    Args:
        A (smatrix): 疎行列
        x (sarray): 疎ベクトル
    Returns:
        sarray: 疎ベクトル
    """
    ax = type(x)(dim = A.M)
    for i in range(len(ax)):
        ax[i] = Array.sum(A.get_vec(i)*x[i])
    return ax

def to_sparse(A:np.ndarray, dtype:str = "COO")->smatrix:
    if dtype == "COO":
        return COO_utils.to_sparse(A)
    else:
        raise NotImplementedError
    
def diag_vec(A:smatrix)->sarray:
    if isinstance(A, coo):
        return COO_utils.diag_vec(A)
    else:
        raise NotImplementedError
    
def diag(A:smatrix)->smatrix:
    if isinstance(A, coo):
        return COO_utils.diag(A)
    else:
        raise NotImplementedError

def rm_diag(A:smatrix)->smatrix:
    if isinstance(A, coo):
        return COO_utils.rm_diag(A)
    else:
        raise NotImplementedError
    
def triu(A:smatrix, k:int = 0)->smatrix:
    if isinstance(A, coo):
        return COO_utils.triu(A)
    else:
        raise NotImplementedError

def tril(A:smatrix, k:int = 0)->smatrix:
    if isinstance(A, coo):
        return COO_utils.tril(A)
    else:
        raise NotImplementedError