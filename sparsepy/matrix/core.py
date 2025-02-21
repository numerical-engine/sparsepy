import numpy as np
import sys
from copy import deepcopy
from sparsepy.vector import svector, scvector


def matmul(mat1:any, mat2:any)->any:
    assert mat1.shape[1] == mat2.shape[0]
    assert type(mat1) == type(mat2)

    output = type(mat1)(mat1.shape[0], mat2.shape[1])
    for r in range(output.shape[0]):
        for c in range(output.shape[1]):
            vec1 = mat1.get_vec(r)
            vec2 = mat2.get_vec(c, 1) if mat2.dtype == float else (mat2.get_vec(c, 1)).conjugate
            output[r, c] = vec1@vec2
        
    return output



class smatrix_meta:
    dtype = float
    def __init__(self, M:int, N:int)->None:
        self.M = M; self.N = N
    @property
    def shape(self)->tuple[int]:
        return (self.M, self.N)
    def __len__(self)->int:
        return self.M
    @property
    def num_nonzero(self)->int:
        raise NotImplementedError
    def to_numpy(self)->np.ndarray:
        raise NotImplementedError
    def where(self, r:int, c:int)->int:
        raise NotImplementedError
    def __setitem__(self, rc:tuple[int], val:any)->None:
        raise NotImplementedError
    def __getitem__(self, rc:tuple[int], val:any)->None:
        raise NotImplementedError
    def __neg__(self)->any:
        raise NotImplementedError
    def __add__(self, another)->any:
        raise NotImplementedError
    def __sub__(self, another)->any:
        return self.__add__(-another)
    def __mul__(self, v:float)->any:
        raise NotImplementedError
    def __rmul__(self, v:float)->any:
        return self.__mul__(v)
    def __truediv__(self, v:float)->any:
        raise NotImplementedError
    def get_vec(self, i:int, axis:int = 0)->svector:
        raise NotImplementedError
    def __matmul__(self, another)->any:
        return matmul(self, another)