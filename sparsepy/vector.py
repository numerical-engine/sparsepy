import numpy as np
from copy import deepcopy
import sys

class svector:
    dtype = float
    def __init__(self, dim:int, indice:np.ndarray = None, values:np.ndarray = None)->None:
        self.dim = dim
        self.indice = np.array([], dtype = int) if indice is None else deepcopy(indice)
        self.values = np.array([], dtype = self.dtype) if values is None else deepcopy(values)

        assert len(self.indice) == len(self.values)
    
    def __len__(self)->int:
        return self.dim
    
    @property
    def num_nonzero(self)->int:
        return len(self.indice)
    
    def to_numpy(self)->np.ndarray:
        array = np.zeros(self.dim, dtype = self.dtype)
        for idx, val in zip(self.indice, self.values):
            array[idx] = val
        return array
    
    def where(self, i:int)->int:
        assert i < self.dim
        index = np.where(self.indice == i)[0]
        assert len(index) <= 1
        return None if len(index) == 0 else index[0]
    
    def __setitem__(self, jdx:int, val:any)->None:
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
    
    def __getitem__(self,  jdx:int)->any:
        idx = self.where(jdx)
        if idx is None:
            return 0.
        else:
            return self.values[idx]
    
    def __neg__(self)->any:
        svec_cpy = deepcopy(self)
        svec_cpy.values = -svec_cpy.values
        return svec_cpy
    
    def __add__(self, another)->any:
        output = deepcopy(self)

        if type(another) != type(self):
            i = np.arange(self.dim, dtype = int)
            v = another*np.ones(self.dim, dtype = self.dtype)
            another = type(self)(dim = self.dim, indice = i, values = v)
        
        assert len(self) == len(another)
        for i, v in zip(another.indice, another.values):
            output[i] += v
        
        return output
    
    def __radd__(self, another)->any:
        return self.__add__(another)
    
    def __sub__(self, another)->any:
        return self.__add__(-another)
    
    def __rsub__(self, another)->any:
        return (-self).__add__(another)
    
    def __mul__(self, v:float)->any:
        output = deepcopy(self)
        output.values *= v

        return output
    
    def __rmul__(self, v:float)->any:
        return self.__mul__(v)
    
    def __truediv__(self, v:float)->any:
        output = deepcopy(self)
        output.values /= v
        return output

    def __matmul__(self, another:any)->float:
        output = 0.
        if len(another) < len(self):
            for i, v in zip(another.indice, another.values):
                output += v*self[i]
        else:
            for i, v in zip(self.indice, self.values):
                output += v*another[i]
        return output

class scvector(svector):
    dtype = complex
    @property
    def conjugate(self)->any:
        output = deepcopy(self)
        output.values = np.conjugate(output.values)

        return output
    
    @property
    def real(self)->svector:
        values_real_wzero = self.values.real
        values_real = np.array([], dtype = svector.dtype)
        indice_real = np.array([], dtype = int)

        for value_real, index in zip(values_real_wzero, self.indice):
            if value_real != 0.:
                indice_real = np.append(indice_real, index)
                values_real = np.append(values_real, value_real)
        
        return svector(dim = self.dim, indice = indice_real, values = values_real)    

    @property
    def imag(self)->svector:
        values_imag_wzero = self.values.imag
        values_imag = np.array([], dtype = svector.dtype)
        indice_imag = np.array([], dtype = int)

        for value_imag, index in zip(values_imag_wzero, self.indice):
            if value_imag != 0.:
                indice_imag = np.append(indice_imag, index)
                values_imag = np.append(values_imag, value_imag)
        
        return svector(dim = self.dim, indice = indice_imag, values = values_imag)
    
    def __matmul__(self, another)->complex:
        output = 0.
        another_conjugate = another.conjugate

        if len(another_conjugate) < len(self):
            for i, v in zip(another_conjugate.indice, another_conjugate.values):
                output += v*self[i]
        else:
            for i, v in zip(self.indice, self.values):
                output += v*another_conjugate[i]
        return output