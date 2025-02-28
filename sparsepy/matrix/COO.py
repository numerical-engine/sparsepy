from sparsepy.core import smatrix, sarray
import numpy as np
from copy import deepcopy
from typing import Self

class coo(smatrix):
    """COO行列

    Attributes:
        M (int): 行数
        N (int): 列数
        rows (np.ndarray): 非ゼロ要素の行インデックス番号
        columns (np.ndarray): 非ゼロ要素の列インデックス番号
        values (np.ndarray): 非ゼロ要素値
    """
    def __init__(self, M:int, N:int, rows:np.ndarray = None, columns:np.ndarray = None, values:np.ndarray = None)->None:
        super().__init__(M, N)
        self.rows = np.array([], dtype = int) if rows is None else deepcopy(rows)
        self.columns = np.array([], dtype = int) if columns is None else deepcopy(columns)
        self.values = np.array([], dtype = float) if columns is None else deepcopy(values)
    
    @property
    def num_nonzero(self)->int:
        return len(self.rows)
    
    def to_numpy(self)->np.ndarray:
        matrix = np.zeros(self.shape, float)
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
    
    def __getitem__(self, rc:tuple[int])->float:
        assert len(rc) == 2
        r, c = rc
        idx = self.where(r, c)
        if idx is None:
            return 0.
        else:
            return self.values[idx]
    
    def __neg__(self)->Self:
        svec_cpy = deepcopy(self)
        svec_cpy.values = -svec_cpy.values
        return svec_cpy
    
    def __add__(self, another:Self)->Self:
        assert type(self) == type(another)
        assert self.shape == another.shape
        output = deepcopy(self)

        for r, c, v in zip(another.rows, another.columns, another.values):
            output[r, c] += v
        
        return output
    
    def __mul__(self, v:float)->Self:
        output = deepcopy(self)
        output.values *= v

        return output
    
    def __truediv__(self, v:float)->Self:
        output = deepcopy(self)
        output.values /= v
        return output
    
    def get_vec(self, i:int, axis:int = 0)->sarray:
        assert axis < 2

        if axis == 0:
            nonzero_idx = np.where(self.rows == i)[0]
            indice = np.array([self.columns[idx] for idx in nonzero_idx], dtype = int)
            dim = self.N
        else:
            nonzero_idx = np.where(self.columns == i)[0]
            indice = np.array([self.rows[idx] for idx in nonzero_idx], dtype = int)
            dim = self.M
        values = np.array([self.values[idx] for idx in nonzero_idx], dtype = float)

        return sarray(dim = dim, indice = indice, values = values)