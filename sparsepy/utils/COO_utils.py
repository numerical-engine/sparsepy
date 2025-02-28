import numpy as np
from sparsepy.core import sarray
from sparsepy.matrix.COO import coo
import sys

def to_sparse(A:np.ndarray)->coo:
    rows = np.array([], dtype = int)
    columns = np.array([], dtype = int)
    values = np.array([], dtype = float)
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            v = A[r, c]
            if v != 0.:
                rows = np.append(rows, r)
                columns = np.append(columns, c)
                values = np.append(values, v)

    return coo(M = A.shape[0], N = A.shape[1], rows = rows, columns = columns, values = values)


def diag_vec(A:coo)->sarray:
    assert A.M == A.N
    d = sarray(dim = A.M)

    pos = np.where((A.rows == A.columns))[0]
    indice = A.rows[pos]
    values = A.values[pos]

    d = sarray(dim = A.M, indice = indice, values = values)
    return d

def diag(A:coo)->coo:
    assert A.M == A.N

    pos = np.where((A.rows == A.columns))[0]
    indice = A.rows[pos]
    values = A.values[pos]

    D = coo(M = A.M, N = A.N, rows = indice, columns = indice, values = values)
    return D

def rm_diag(A:coo)->coo:
    assert A.M == A.N

    pos = np.where((A.rows != A.columns))[0]
    rows = A.rows[pos]
    columns = A.columns[pos]
    values = A.values[pos]

    D = coo(M = A.M, N = A.N, rows = rows, columns = columns, values = values)
    return D

def triu(A:coo, k:int = 0)->coo:
    pos = np.where((A.columns >= (A.rows + k)))[0]
    rows = A.rows[pos]
    columns = A.columns[pos]
    values = A.values[pos]

    D = coo(M = A.M, N = A.N, rows = rows, columns = columns, values = values)
    return D

def tril(A:coo, k:int = 0)->coo:
    pos = np.where((A.rows >= (A.columns + k)))[0]
    rows = A.rows[pos]
    columns = A.columns[pos]
    values = A.values[pos]

    D = coo(M = A.M, N = A.N, rows = rows, columns = columns, values = values)
    return D