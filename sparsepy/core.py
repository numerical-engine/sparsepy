import numpy as np
import sys
from copy import deepcopy

class smatrix_meta:
    """Abstract class for sparse matrix
    """
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
    
    def get_vec(self, i:int, axis:int = 0)->any:
        print("here")
        sys.exit()
    
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
    
    def __matmul__(self, another)->float:
        assert self.shape[1] == another.shape[0]