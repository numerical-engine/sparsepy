import numpy as np
from copy import deepcopy
import sys
from typing import Self


class sarray:
    """疎ベクトルクラス

    Attributes:
        dim (int): ベクトルの次元数
        indice (np.ndarray, optional): 非ゼロ要素のインデックス番号。Noneの場合ゼロベクトル。
        values (np.ndarray, optional): 非ゼロ要素の値。Noneの場合ゼロベクトル。
    """
    def __init__(self, dim:int, indice:np.ndarray = None, values:np.ndarray = None)->None:
        self.dim = dim
        self.indice = np.array([], dtype = int) if indice is None else deepcopy(indice)
        self.values = np.array([], dtype = float) if values is None else deepcopy(values)

        assert len(self.indice) == len(self.values)
    
    def __len__(self)->int:
        return self.dim
    
    @property
    def num_nonzero(self)->int:
        """非ゼロ要素の個数を返す。

        Returns:
            int: 非ゼロ要素の個数
        """
        return len(self.indice)
    
    def to_numpy(self)->np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
        """
        array = np.zeros(self.dim, dtype = float)
        for idx, val in zip(self.indice, self.values):
            array[idx] = val
        return array
    
    def where(self, i:int)->int:
        """indice[j] == iとなるインデックス番号を返す。

        Args:
            i (int): インデックス番号i
        Returns:
            int: インデックス番号j
        """
        assert i < self.dim
        index = np.where(self.indice == i)[0]
        assert len(index) <= 1
        return None if len(index) == 0 else index[0]
    
    def __setitem__(self, jdx:int, val:float)->None:
        idx = self.where(jdx)
        
        if idx is None:
            if val != 0.:
                self.indice = np.append(self.indice, jdx)
                self.values = np.append(self.values, val)
        else:
            if val == 0.:
                self.indice = np.delete(self.indice, idx)
                self.values = np.delete(self.values, idx)
            else:
                self.values[idx] = val
    
    def __getitem__(self,  jdx:int)->float:
        idx = self.where(jdx)
        if idx is None:
            return 0.
        else:
            return self.values[idx]
    
    def __neg__(self)->Self:
        svec_cpy = deepcopy(self)
        svec_cpy.values = -svec_cpy.values
        return svec_cpy
    
    def __add__(self, another)->Self:
        output = deepcopy(self)

        if type(another) != type(self):
            i = np.arange(self.dim, dtype = int)
            v = another*np.ones(self.dim, dtype = float)
            another = type(self)(dim = self.dim, indice = i, values = v)
        
        assert len(self) == len(another)
        for i, v in zip(another.indice, another.values):
            output[i] += v
        
        return output
    
    def __radd__(self, another)->Self:
        return self.__add__(another)
    
    def __sub__(self, another)->Self:
        return self.__add__(-another)
    
    def __rsub__(self, another)->Self:
        return (-self).__add__(another)
    
    def __mul__(self, v:float)->Self:
        output = deepcopy(self)
        output.values *= v

        return output
    
    def __rmul__(self, v:float)->Self:
        return self.__mul__(v)
    
    def __truediv__(self, another:any)->Self:
        output = deepcopy(self)
        if type(self) == type(another):
            for idx in output.indice:
                output[idx] /= another[idx]
        else:
            output = deepcopy(self)
            output.values /= another
        
        return output

    def __matmul__(self, another:any)->float:
        output = 0.
        if len(another) < len(self):
            for i, v in zip(another.indice, another.values):
                output += v*self[i]
        else:
            for i, v in zip(self.indice, self.values):
                output += v*another[i]
        return float(output)


class smatrix:
    """疎行列のための抽象クラス

    Attributes:
        M (int): 行数
        N (int): 列数
    """
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
    def __setitem__(self, rc:tuple[int], val:float)->None:
        raise NotImplementedError
    def __getitem__(self, rc:tuple[int], val:float)->None:
        raise NotImplementedError
    def __neg__(self)->Self:
        raise NotImplementedError
    def __add__(self, another:any)->Self:
        raise NotImplementedError
    def __sub__(self, another:any)->Self:
        return self.__add__(-another)
    def __mul__(self, v:float)->Self:
        raise NotImplementedError
    def __rmul__(self, v:float)->Self:
        return self.__mul__(v)
    def __truediv__(self, v:float)->Self:
        raise NotImplementedError
    def get_vec(self, i:int, axis:int = 0)->sarray:
        raise NotImplementedError

    def __matmul__(self, another:Self)->Self:
        assert self.shape[1] == another.shape[0]
        assert type(self) == type(another)

        output = type(self)(self.shape[0], self.shape[1])
        for r in range(output.shape[0]):
            for c in range(output.shape[1]):
                vec1 = self.get_vec(r)
                vec2 = self.get_vec(c, 1)
                output[r, c] = vec1@vec2
            
        return output