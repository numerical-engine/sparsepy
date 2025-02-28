import numpy as np
from sparsepy.core import sarray, smatrix
from sparsepy import utils
from sparsepy.utils.Matrix import Ax
import sys

def jacob(A:smatrix, b:sarray, x:sarray = None, w:float = 1., maxiter:int = 1000, rtol:float = 0., ttol:float = 0.)->tuple:
    """ヤコビ法による線形ソルバー

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
    
    D = utils.Matrix.diag_vec(A)
    R = utils.Matrix.rm_diag(A)

    residuals = [utils.Array.norm(b - Ax(A, x))]
    for _ in range(maxiter):
        x = w*((b - Ax(R, x)) / D) + (1. - w)*x
        residuals.append(utils.Array.norm(b - Ax(A, x)))

        if residuals[-1]/residuals[0] < rtol:
            return x, residuals
        if residuals[-1] < ttol:
            return x, residuals
    
    return x, residuals