# pylint: disable=C0103,C0111,C0301
import math
import casadi as ca

from config import *

__all__ = [
    'constructBetaT','constructEi','constructFi','constructF0',
    'constructEM','constructM','constructB','constructBBTint',
    'create_L1_function','create_T_function','constructCkptMat','constructNPiecesCkptMat','create_softmax_function'
]


def constructBetaT(t, rank:int):
    ''' 构造特定时间的β(t) '''
    beta = SYM_TYPE(NCOFF, 1)
    for i in range(rank, NCOFF):
        if not isinstance(t, int|float) or t!=0 or i-rank==0:
            beta[i] = math.factorial(i)/math.factorial(i-rank) * t**(i-rank)
    return beta

def constructEi(T):
    ''' 构造M矩阵中的Ei(2s*2s)=[β(T), β(T), ..., β(T)^(2s-2)] '''
    Ei = SYM_TYPE(NCOFF, NCOFF)
    Ei[0, :] = constructBetaT(T, 0)
    for i in range(1, NCOFF):
        Ei[i, :] = constructBetaT(T, i-1)
    return Ei

def constructFi():
    ''' 构造M矩阵中的Fi(2s*2s)=[0, -β(0), ..., β(0)^(2s-2)] '''
    Fi = SYM_TYPE(NCOFF, NCOFF)
    for i in range(1, NCOFF):
        Fi[i, :] = -constructBetaT(0, i-1)
    return Fi

def constructF0():
    ''' 构造M矩阵中的F0(s*2s)=[β(0), ..., β(0)^(s-1)] '''
    F0 = SYM_TYPE(S, NCOFF)      # 端点约束
    for i in range(S):
        F0[i, :] = constructBetaT(0, i)
    return F0

def constructEM(T):
    ''' 构造M矩阵中的E0(s*2s)=[β(T), ..., β(T)^(s-1)] '''
    E0 = SYM_TYPE(S, NCOFF)      # 端点约束
    for i in range(S):
        E0[i, :] = constructBetaT(T, i)
    return E0

def constructM(pieceT, num_pieces):
    ''' 构造矩阵M=[
        [F0,    0,      0,      0,    ...,    ],
        [E1,   F1,      0,      0,    ...,    ],
        [ 0,   E2,     F2,      0,    ...,    ],
        ...
    ]
    '''
    M = SYM_TYPE(num_pieces*NCOFF, num_pieces*NCOFF)
    # F0 = SYM_TYPE.sym('F0', 3, 6)
    # EM = SYM_TYPE.sym('EM', 3, 6)
    M[0:S, 0:NCOFF] = constructF0()         # 3*6
    M[-S:, -NCOFF:] = constructEM(T=pieceT) # 3*6
    for i in range(1, num_pieces):
        # Fi = SYM_TYPE.sym(f'F{i}', 6, 6)
        # Ei = SYM_TYPE.sym(f'E{i}', 6, 6)
        M[(i-1)*NCOFF+S:i*NCOFF+S, (i-1)*NCOFF:i*NCOFF] = constructEi(pieceT)
        M[(i-1)*NCOFF+S:i*NCOFF+S, i*NCOFF:(i+1)*NCOFF] = constructFi()
    return M

def constructB(state0, stateT, mid_pos, num_pieces):
    ''' 构造右端路径点约束B矩阵'''
    B = SYM_TYPE(num_pieces*NCOFF, NDIM)
    B[0:S,:] = state0                     # 起点状态
    B[-S:,:] = stateT                       # 终点状态
    for i in range(1, num_pieces):
        B[(i-1)*NCOFF+S, :] = mid_pos[i-1, :]      # 设置中间路径点
    return B

def constructBBTint(pieceT, rank):
    ''' c^T*(∫β*β^T*dt)*c ==> (2, NCOFF) * (NCOFF, NCOFF) * (NCOFF, 2) '''
    bbint = SYM_TYPE(NCOFF, NCOFF)
    beta = constructBetaT(pieceT, rank)
    for i in range(NCOFF):
        for j in range(NCOFF):
            if i+j-2*rank < 0: continue
            coff = 1 / (i+j-2*rank+1)
            bbint[i, j] = coff * beta[i] * beta[j] * pieceT
    return bbint

def create_L1_function():
    a0 = 1e-1
    x = ca.SX.sym('x')

    f1 = 0  # x <= 0
    f2 = -1/(2*a0**3)*x**4 + 1/a0**2*x**3  # 0 < x <= a0
    f3 = x - a0/2  # a0 < x

    L1 = ca.if_else(x <= 0, f1,
         ca.if_else(x <= a0, f2, f3))

    L1_func = ca.Function('L1', [x], [L1])

    return L1_func

def create_T_function():
    tau = ca.SX.sym('tau')

    f1 = 0.5 * tau**2 + tau + 1  # tau > 0
    f2 = (tau**2 - 2*tau + 2)/2  # tau <= 0

    Tf = ca.if_else(tau > 0, f1, f2)

    T_func = ca.Function('T', [tau], [tau**2+0.01])

    return T_func

def create_softmax_function():
    x = ca.SX.sym('x')
    return ca.Function('sigmoid', [x], [1 / (1 + ca.exp(-x))])


def constructCkptMat(pieceT, num_ckpt:int, rank:int):
    ''' 构造单个piece的CKPT检查矩阵 '''
    ckpt_mat = SYM_TYPE(NCOFF, num_ckpt)
    ckpt_frac = [(i+1)/(num_ckpt+1) for i in range(num_ckpt)]
    ckpt_ts = ckpt_frac * pieceT

    for i in range(num_ckpt):
        ckpt_mat[:, i] = constructBetaT(t=ckpt_ts[i], rank=rank)
    return ckpt_mat

def constructNPiecesCkptMat(pieceT, rank:int, nckpt:int, npiece:int):
    ''' 构造整条轨迹的ckpt检查矩阵[NPIECE*NCOFF, NPIECE*NCKPT] '''
    ckpt_mat = SYM_TYPE(npiece*NCOFF, npiece*nckpt)
    for i in range(npiece):
        ckpt_mat[i*NCOFF:(i+1)*NCOFF, i*nckpt:(i+1)*nckpt] = constructCkptMat(pieceT=pieceT, num_ckpt=nckpt, rank=rank)
    return ckpt_mat
    # return ca.repmat(constructCkptMat(pieceT=pieceT, num_ckpt=nckpt, rank=rank), npiece, 1)