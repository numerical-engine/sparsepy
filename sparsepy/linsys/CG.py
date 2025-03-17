import numpy as np
from copy import deepcopy
from sparsepy.core import sarray, smatrix
from sparsepy import utils
from sparsepy.linsys import linsys_utils
from sparsepy.utils.Matrix import Ax
import sys


def cg(A:smatrix, b:sarray, x:sarray = None, w:float = 1., maxiter:int = 1000, rtol:float = 1e-5, ttol:float = 0.)->tuple:
    """CG法による線形ソルバー

    Args:
        A (smatrix): 係数行列
        b (sarray): 右辺ベクトル
        x (sarray, optional): 初期候補解。Noneの場合ゼロベクトル。
        w (float): 緩和係数。
        maxiter (int, optional): 収束条件。最大反復回数。
        rtol (float, optional): 収束条件。残差比。
        ttol (float, optional): 収束条件。残差。
    Returns:
        sarray: 解。
    """

    if x is None:
        x = sarray(dim = b.dim)
    
    r = b - Ax(A, x)
    p = deepcopy(r)

    residuals = [linsys_utils.get_residual(A, b, x)]

    for _ in range(maxiter):
        Ap = Ax(A, p)
        alpha = (residuals[-1]**2)/(p@Ap)
        x += w*alpha*p
        r -= alpha*Ap

        residuals.append(utils.Array.norm(r))

        if linsys_utils.check_termination(residuals, rtol, ttol):
            return x, residuals
        
        beta = (residuals[-1]**2)/(residuals[-2]**2)
        p = r + beta*p

    return x, residuals