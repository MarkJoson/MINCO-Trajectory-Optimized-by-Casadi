# pylint: disable=C0103,C0111,C0301
import math
import casadi as ca
import numpy as np

from config import *

__all__ = [
    'constructBetaT','constructEi','constructFi','constructF0',
    'constructEM','constructM','constructB','constructBBTint',
    'L1_func','tau2T_func','constructCkptMat','constructNPiecesCkptMat','softmax_func',
    'create_poly_eval_func', 'create_traj_eval_func'
]


def constructBetaT(t, rank:int):
    ''' 构造特定时间的β(t) '''
    beta = ca.SX(NCOFF, 1)
    for i in range(rank, NCOFF):
        if not isinstance(t, int|float) or t!=0 or i-rank==0:
            beta[i,0] = math.factorial(i)/math.factorial(i-rank) * t**(i-rank)
    return beta

def constructEi(T):
    ''' 构造M矩阵中的Ei(2s*2s)=[β(T), β(T), ..., β(T)^(2s-2)] '''
    Ei = ca.SX(NCOFF, NCOFF)
    Ei[0, :] = constructBetaT(T, 0)
    for i in range(1, NCOFF):
        Ei[i, :] = constructBetaT(T, i-1)
    return Ei

def constructFi():
    ''' 构造M矩阵中的Fi(2s*2s)=[0, -β(0), ..., β(0)^(2s-2)] '''
    Fi = ca.SX(NCOFF, NCOFF)
    for i in range(1, NCOFF):
        Fi[i, :] = -constructBetaT(0, i-1)
    return Fi

def constructF0():
    ''' 构造M矩阵中的F0(s*2s)=[β(0), ..., β(0)^(s-1)] '''
    F0 = ca.SX(S, NCOFF)      # 端点约束
    for i in range(S):
        F0[i, :] = constructBetaT(0, i)
    return F0

def constructEM(T):
    ''' 构造M矩阵中的E0(s*2s)=[β(T), ..., β(T)^(s-1)] '''
    E0 = ca.SX(S, NCOFF)      # 端点约束
    for i in range(S):
        E0[i, :] = constructBetaT(T, i)
    return E0

def constructM(pieceT, num_pieces, SYMT=SYM_TYPE):
    ''' 构造矩阵M=[
        [F0,    0,      0,      0,    ...,    ],
        [E1,   F1,      0,      0,    ...,    ],
        [ 0,   E2,     F2,      0,    ...,    ],
        ...
    ]
    '''
    M = ca.SX(num_pieces*NCOFF, num_pieces*NCOFF)
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
    B = ca.SX(num_pieces*NCOFF, NDIM)
    B[0:S,:] = state0                     # 起点状态
    B[-S:,:] = stateT                       # 终点状态
    for i in range(1, num_pieces):
        B[(i-1)*NCOFF+S, :] = mid_pos[i-1, :]      # 设置中间路径点
    return B

def constructBBTint(pieceT, rank):
    ''' c^T*(∫β*β^T*dt)*c ==> (2, NCOFF) * (NCOFF, NCOFF) * (NCOFF, 2) '''
    bbint = ca.SX(NCOFF, NCOFF)
    beta = constructBetaT(pieceT, rank)
    for i in range(NCOFF):
        for j in range(NCOFF):
            if i+j-2*rank < 0: continue
            coff = 1 / (i+j-2*rank+1)
            bbint[i, j] = coff * beta[i,0] * beta[j,0] * pieceT
    return bbint

def L1_func(x):
    a0 = 1e-1

    f1 = 0  # x <= 0
    f2 = -1/(2*a0**3)*x**4 + 1/a0**2*x**3  # 0 < x <= a0
    f3 = x - a0/2  # a0 < x

    L1 = ca.if_else(x <= 0, f1,
         ca.if_else(x <= a0, f2, f3))

    return L1

def tau2T_func(tau):

    # f1 = 0.5 * tau**2 + tau + 1  # tau > 0
    # f2 = (tau**2 - 2*tau + 2)/2  # tau <= 0
    # Tf = ca.if_else(tau > 0, f1, f2)

    return tau**2+1e-2

def softmax_func(x):
    return 1 / (1 + ca.exp(-x))


def constructCkptMat(pieceT, num_ckpt:int, rank:int):
    ''' 构造单个piece的CKPT检查矩阵 '''
    ckpt_mat = ca.SX(NCOFF, num_ckpt)
    ckpt_frac = np.array([(i+1)/(num_ckpt+1) for i in range(num_ckpt)])
    ckpt_ts = ckpt_frac * pieceT

    for i in range(num_ckpt):
        ckpt_mat[:, i] = constructBetaT(t=ckpt_ts[i], rank=rank)
    return ckpt_mat

def constructNPiecesCkptMat(pieceT, rank:int, nckpt:int, npiece:int):
    ''' 构造整条轨迹的ckpt检查矩阵[NPIECE*NCOFF, NPIECE*NCKPT] '''
    ckpt_mat = ca.SX(npiece*NCOFF, npiece*nckpt)
    for i in range(npiece):
        ckpt_mat[i*NCOFF:(i+1)*NCOFF, i*nckpt:(i+1)*nckpt] = constructCkptMat(pieceT=pieceT, num_ckpt=nckpt, rank=rank)
    return ckpt_mat


def create_poly_eval_func(n_piece, n_drawpt):
    '''多项式评估函数，输入多项式系数，输出采样点'''
    T = ca.SX.sym('T') # type: ignore
    coff = ca.SX.sym('coff', NCOFF*n_piece, NDIM) # type: ignore

    pos_ckpt_mat = constructNPiecesCkptMat(pieceT=T, rank=0, nckpt=n_drawpt, npiece=n_piece)
    vel_ckpt_mat = constructNPiecesCkptMat(pieceT=T, rank=1, nckpt=n_drawpt, npiece=n_piece)
    acc_ckpt_mat = constructNPiecesCkptMat(pieceT=T, rank=2, nckpt=n_drawpt, npiece=n_piece)
    jerk_ckpt_mat = constructNPiecesCkptMat(pieceT=T, rank=3, nckpt=n_drawpt, npiece=n_piece)
    snap_ckpt_mat = constructNPiecesCkptMat(pieceT=T, rank=4, nckpt=n_drawpt, npiece=n_piece)

    pos_ckpts = pos_ckpt_mat.T @ coff
    vel_ckpts = vel_ckpt_mat.T @ coff
    acc_ckpts = acc_ckpt_mat.T @ coff
    jerk_ckpts = jerk_ckpt_mat.T @ coff
    snap_ckpts = snap_ckpt_mat.T @ coff

    vels_sqsum = ca.sum2(vel_ckpts ** 2)

    numerator = (vel_ckpts[:,0]*acc_ckpts[:,1] - vel_ckpts[:,1]*acc_ckpts[:,0])**2
    denominator = vels_sqsum**3
    curvature_sq_ckpts = numerator / (denominator+1e-10)

    return ca.Function(
        'poly_eval',
        [T, coff],
        [pos_ckpts, vel_ckpts, acc_ckpts, curvature_sq_ckpts, jerk_ckpts, snap_ckpts],
        ["T", "coff"],
        ["pos_ckpts", "vel_ckpts", "acc_ckpts", "curvature_sq_ckpts", "jerk_ckpts", "snap_ckpts"]
    )


def create_traj_eval_func(n_piece, n_drawpt, n_mid_pos):
    ''' 轨迹（多条piece）评估函数，输入起始点和中止点，输出采样点'''
    T = ca.SX.sym('T') # type: ignore
    state0 = ca.SX.sym('state0', S, NDIM)                # 起始坐标，行向量 # type: ignore
    stateT = ca.SX.sym('stateT', S, NDIM)                # 结束坐标，行向量 # type: ignore
    mid_pos = ca.SX.sym('mid_pos', n_mid_pos, NDIM)      # 中间坐标，行向量 # type: ignore

    poly_eval_fn = create_poly_eval_func(n_piece=n_piece, n_drawpt=n_drawpt)

    M = constructM(pieceT=T, num_pieces=n_piece)
    B = constructB(state0=state0, stateT=stateT, mid_pos=mid_pos, num_pieces=n_piece)
    c = ca.solve(M, B)

    result = poly_eval_fn.call({"T":T, "coff":c})

    return ca.Function(
        'mmp_eval',
        [T, state0, stateT, mid_pos],
        [c, *result.values()],
        ["T","state0","stateT","mid_pos"],
        ["coff", *result.keys()]
    )
