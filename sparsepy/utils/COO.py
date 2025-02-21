import numpy as np
from sparsepy import coo, ccoo

def to_sparse(A:np.ndarray, dtype:type)->any:
    rows = np.array([], dtype = int)
    columns = np.array([], dtype = int)
    values = np.array([], dtype = dtype)
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            v = A[r, c]
            if v != 0.:
                rows = np.append(rows, r)
                columns = np.append(columns, c)
                values = np.append(values, v)
    
    if dtype == float:
        return coo(M = A.shape[0], N = A.shape[1], rows = rows, columns = columns, values = values)
    else:
        return ccoo(M = A.shape[0], N = A.shape[1], rows = rows, columns = columns, values = values)