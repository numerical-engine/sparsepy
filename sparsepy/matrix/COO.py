from sparsepy.matrix.core import smatrix_meta
import numpy as np
from copy import deepcopy
from sparsepy.vector import svector, scvector
import sys


def get_vec(mat:any, i:int, axis:int = 0)->any:
    assert axis < 2

    if axis == 0:
        nonzero_idx = np.where(mat.rows == i)[0]
        indice = np.array([mat.columns[idx] for idx in nonzero_idx], dtype = int)
        dim = mat.N
    else:
        nonzero_idx = np.where(mat.columns == i)[0]
        indice = np.array([mat.rows[idx] for idx in nonzero_idx], dtype = int)
        dim = mat.M
    values = np.array([mat.values[idx] for idx in nonzero_idx], dtype = mat.dtype)

    if mat.dtype == float:
        return svector(dim = dim, indice = indice, values = values)
    else:
        return scvector(dim = dim, indice = indice, values = values)


class coo(smatrix_meta):
    def __init__(self, M:int, N:int, rows:np.ndarray = None, columns:np.ndarray = None, values:np.ndarray = None)->None:
        super().__init__(M, N)
        self.rows = np.array([], dtype = int) if rows is None else deepcopy(rows)
        self.columns = np.array([], dtype = int) if columns is None else deepcopy(columns)
        self.values = np.array([], dtype = self.dtype) if columns is None else deepcopy(values)
    
    @property
    def num_nonzero(self)->int:
        return len(self.rows)
    
    def to_numpy(self)->np.ndarray:
        matrix = np.zeros(self.shape, dtype = self.dtype)
        for r, c, v in zip(self.rows, self.columns, self.values):
            matrix[r, c] = v
        return matrix
    
    def where(self, r:int, c:int)->int:
        assert (r < self.M) & (c < self.N)
        index = None
        idx = 0
        for row, col in zip(self.rows, self.columns):
            if (r == row) & (c == col):
                index = idx
                break
            idx += 1
        return index
    
    def __setitem__(self, rc:tuple[int], val:float)->None:
        assert len(rc) == 2
        r, c = rc
        idx = self.where(r, c)
        
        if idx is None:
            if val != 0.:
                self.rows = np.append(self.rows, r)
                self.columns = np.append(self.columns, c)
                self.values = np.append(self.values, val)
        else:
            if val == 0.:
                self.rows = np.delete(self.rows, idx)
                self.columns = np.delete(self.columns, idx)
                self.values = np.delete(self.values, idx)
            else:
                self.values[idx] = val
    
    def __getitem__(self, rc:tuple[int])->None:
        assert len(rc) == 2
        r, c = rc
        idx = self.where(r, c)
        if idx is None:
            return 0.
        else:
            return self.values[idx]
    
    def __neg__(self)->any:
        svec_cpy = deepcopy(self)
        svec_cpy.values = -svec_cpy.values
        return svec_cpy
    
    def __add__(self, another)->any:
        assert type(self) == type(another)
        assert self.shape == another.shape
        output = deepcopy(self)

        for r, c, v in zip(another.rows, another.columns, another.values):
            output[r, c] += v
        
        return output
    
    def __mul__(self, v:float)->any:
        output = deepcopy(self)
        output.values *= v

        return output
    
    def __truediv__(self, v:float)->any:
        output = deepcopy(self)
        output.values /= v
        return output
    
    def get_vec(self, i:int, axis:int = 0)->svector:
        return get_vec(self, i, axis)


class ccoo(coo):
    dtype = complex
    @property
    def conjugate(self)->any:
        output = deepcopy(self)
        output.values = np.conjugate(output.values)

        return output
    
    @property
    def real(self)->coo:
        values_real_wzero = self.values.real
        values_real = np.array([], dtype = coo.dtype)
        rows_real = np.array([], dtype = int)
        columns_real = np.array([], dtype = int)

        for value_real, row, col in zip(values_real_wzero, self.rows, self.columns):
            if value_real != 0.:
                rows_real = np.append(rows_real, row)
                columns_real = np.append(columns_real, col)
                values_real = np.append(values_real, value_real)
        
        return coo(M = self.M, N = self.N, rows = rows_real, columns = columns_real, values = values_real)
    
    @property
    def imag(self)->coo:
        values_imag_wzero = self.values.imag
        values_imag = np.array([], dtype = coo.dtype)
        rows_imag = np.array([], dtype = int)
        columns_imag = np.array([], dtype = int)

        for value_imag, row, col in zip(values_imag_wzero, self.rows, self.columns):
            if value_imag != 0.:
                rows_imag = np.append(rows_imag, row)
                columns_imag = np.append(columns_imag, col)
                values_imag = np.append(values_imag, value_imag)
        
        return coo(M = self.M, N = self.N, rows = rows_imag, columns = columns_imag, values = values_imag)