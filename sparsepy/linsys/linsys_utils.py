from sparsepy.utils.Matrix import Ax
from sparsepy.core import sarray, smatrix
from sparsepy import utils

def get_residual(A:smatrix, b:sarray, x:sarray)->float:
    return utils.Array.norm(b - Ax(A, x))

def check_termination(residuals:list[float], rtol:float = 0., ttol:float = 0.)->bool:
    flag = (residuals[-1]/residuals[0] < rtol) + (residuals[-1] < ttol)
    return flag