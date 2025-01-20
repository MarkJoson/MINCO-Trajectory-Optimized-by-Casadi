# pylint: disable=C0103,C0111,C0301 W0718,W0401
import math
from typing import Dict, List, Tuple

import numpy as np
import casadi as ca
from scipy import sparse
from evaluate import create_visualization
from debug_casadi import print_matrix, print_structured_matrix
import control
import scipy.linalg as sl

from config import *
from toolbox import L1_func, create_poly_eval_func

np.set_printoptions(precision=50)

# 轨迹优化参数
NTRAJ     = 1         # 轨迹条数
NPIECE    = 1         # 5阶段轨迹曲线
NMIDPT    = NTRAJ*NPIECE-1      # 中间路径点（优化变量）个数
NCKPT     = 5        # 检查点个数
NDRAW_PT  = 50

PIECE_T  = 0.1
RATIO = 0.1

# 可行性参数
max_vel_sq = 1
max_acc_sq = 1
max_cur_sq = 1.5

# 优化权重
weight_vel = 100.
weight_acc = 100.0
weight_cur = 0.0

SYM_TYPE = ca.MX

def constructBetaT(t, rank:int):
    ''' 构造特定时间的β(t) '''
    beta = ca.DM(NCOFF, 1)
    for i in range(rank, NCOFF):
        if not isinstance(t, int|float) or t!=0 or i-rank==0:
            beta[i,0] = math.factorial(i)/math.factorial(i-rank) * t**(i-rank)
    return beta

def constructBBTint(pieceT, rank):
    ''' c^T*(∫β*β^T*dt)*c ==> (2, NCOFF) * (NCOFF, NCOFF) * (NCOFF, 2) '''
    bbint = ca.DM(NCOFF, NCOFF)
    beta = constructBetaT(pieceT, rank)
    for i in range(NCOFF):
        for j in range(NCOFF):
            if i+j-2*rank < 0: continue
            coff = 1 / (i+j-2*rank+1)
            bbint[i, j] = coff * beta[i,0] * beta[j,0] * pieceT
    return bbint

def constructCkptMat(pieceT, num_ckpt:int, rank:int):
    ''' 构造单个piece的CKPT检查矩阵 '''
    ckpt_mat = ca.DM(NCOFF, num_ckpt)
    ckpt_frac = np.array([(i+1)/(num_ckpt+1) for i in range(num_ckpt)])
    ckpt_ts = ckpt_frac * pieceT

    for i in range(num_ckpt):
        ckpt_mat[:, i] = constructBetaT(t=ckpt_ts[i], rank=rank)
    return ckpt_mat

def constructNPiecesCkptMat(pieceT, rank:int, nckpt:int, npiece:int):
    ''' 构造整条轨迹的ckpt检查矩阵[NPIECE*NCOFF, NPIECE*NCKPT] '''
    ckpt_mat = ca.DM(npiece*NCOFF, npiece*nckpt)
    for i in range(npiece):
        ckpt_mat[i*NCOFF:(i+1)*NCOFF, i*nckpt:(i+1)*nckpt] = constructCkptMat(pieceT=pieceT, num_ckpt=nckpt, rank=rank)
    return ckpt_mat



def constructMincoM2(pieceT):
    mat_m = ca.DM(NCOFF, NCOFF)
    for i in range(NCOFF-2):
        mat_m[i, :] = constructBetaT(0, i).T

    mat_m[-2, :] = constructBetaT(pieceT, 0).T
    mat_m[-1, :] = constructBetaT(pieceT, 1).T

    mat_supp = ca.DM([[0,0,0,0,0,1]]) @ ca.solve(mat_m, constructBBTint(pieceT=pieceT, rank=S))

    mat_m[-1, :] = mat_supp[-1, :]

    return mat_m


def consturctMatR(pieceT):
    mat_r = ca.DM(NCOFF, NCOFF)
    for i in range(NCOFF):
        mat_r[i, :] = constructBetaT(pieceT, i).T
    return mat_r

def final_pos_to_coff(pos):
    coff = ca.DM(6,2)
    coff[0,:] = pos
    return coff


def state2coff(state):
    mat_r = consturctMatR(pieceT=0)
    coff = mat_r[:, 0:S] @ state
    return ca.DM(coff)

def solve_coff(current_coff, pieceT, end_pos):
    # 计算本段轨迹的起点和终点
    mat_r = ca.DM(consturctMatR(pieceT=pieceT/RATIO))
    mat_s = ca.DM(np.diag([1,1,1,1,0,0]))
    mat_u = ca.DM([[0],[0],[0],[0],[1],[0]])

    Minv = np.linalg.inv(ca.DM(constructMincoM2(pieceT=pieceT)))

    mat_F = Minv @ mat_s @ mat_r
    mat_G = Minv @ mat_u

    # LQR
    Q = constructBBTint(pieceT=pieceT, rank=3)# + np.diag([1,1,1,0,0,0])*3
    R = np.array([[1]])     # 控制权重矩阵
    K, _, _ = control.dlqr(ca.DM(mat_F), ca.DM(mat_G), ca.DM(Q), ca.DM(R))

    Kpp = sl.pinv(mat_G) @ (np.identity(6)-mat_F)+K
    mat_F_stab = mat_F - mat_G @ K
    mat_G_stab = mat_G @ Kpp @ np.array([[1,0,0,0,0,0]]).T

    new_coff = mat_F @ current_coff + mat_G @ end_pos
    return new_coff



def create_obj_with_cons_fn(accept_pos, begin_coff, end_coff):
    ''' 代价函数
    中间点的个数为: NTraj * NPiece - 1
    其中i*NPiece是轨迹i和轨迹i+1的交界
    [i*NPiece, (i+1)*NPiece-1) 一共Npiece-1个点是中间点
    换挡点（切换点）: i*NPIECE-1
    ×  ▲  ▲  ▲  ●  ▲  ▲  ▲  ●  ▲  ▲  ▲  ●  ▲  ▲  ▲  ×
       -  -  -  ↑  -  -  -  ↑  -  -  -  ↑  -  -  -
       0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 '''

    cost =  0

    # accept_pos = SYM_TYPE.sym('accept_pos', NTRAJ, NDIM)
    # begin_coff = SYM_TYPE.sym('end_coff', NCOFF, NDIM)
    # end_coff = SYM_TYPE.sym('end_coff', NCOFF, NDIM)

    weight = 0.01

    current_coff = begin_coff
    for i in range(NTRAJ):
        # 本段轨迹的piece时长
        current_coff = solve_coff(current_coff=current_coff, pieceT=PIECE_T, end_pos=accept_pos[i,:])

        # Soft-Constrain
        vel_ckm = constructNPiecesCkptMat(pieceT=PIECE_T/RATIO, rank=1, nckpt=NCKPT, npiece=NPIECE)
        acc_ckm = constructNPiecesCkptMat(pieceT=PIECE_T/RATIO, rank=2, nckpt=NCKPT, npiece=NPIECE)

        # [NPIECE*NCOFF, NCKPT].T @ [NPIECE*NCOFF, NDIM] -> [NUM_CKPT, NDIM]
        vels = vel_ckm.T @ current_coff
        accs = acc_ckm.T @ current_coff

        vels_sqsum = ca.sum2(vels ** 2)
        accs_sqsum = ca.sum2(accs ** 2)

        numerator = (vels[:,0]*accs[:,1] - vels[:,1]*accs[:,0])**2
        denominator = vels_sqsum**3
        curvature_sq = numerator / (denominator+1e-6)

        con_vel = ca.sum1(L1_func(vels_sqsum - max_vel_sq))
        con_acc = ca.sum1(L1_func(accs_sqsum - max_acc_sq))
        # con_cur = ca.sum1(L1_func(curvature_sq - max_cur_sq))

        # 可行性代价
        cost += weight_vel*con_vel + weight_acc*con_acc# + weight_cur*con_cur

    # 最终距离目标的代价
    # current_coff = SYM_TYPE.sym('current_coff', NCOFF, NDIM)
    # final_diff = (end_coff - current_coff) * SYM_TYPE([[1,0,0,0,0,0]]).T
    final_diff = (constructBetaT(PIECE_T/RATIO,rank=0).T @ current_coff - end_coff[0,0:2])
    # # final_diff = accept_pos - SYM_TYPE([[3,1]])
    cost += 0.0001*ca.trace(final_diff.T@final_diff) #(ca.norm_2(final_diff))

    return cost
    # return ca.Function('cost_with_cons_obj', [accept_pos, begin_coff, end_coff], [cost])

def wrap(cost_fn, accept_pos, begin_coff, end_coff):
    val = cost_fn(accept_pos, begin_coff, end_coff)
    return val

## ^-------------------------------求解----------------------------------####
solver_specific_options = {
    'bonmin': {
        # 'bonmin.algorithm': 'B-BB',
        # 'bonmin.solution_limit': 1
    },
    'ipopt': {
        'ipopt.max_iter': 100,
        'ipopt.tol': 1e-4,
        'ipopt.print_level': 0,
        'ipopt.mu_strategy': 'adaptive',
        'print_time': True
    },
    'knitro': {
        'knitro.algorithm': 1,
        'knitro.mir_maxiter': 1000
    },
    'sqpmethod': {
        'max_iter': 1000,
        'print_iteration': True,
        'print_header': True,
        'print_status': True,
        'tol_du': 1e-6,
        'tol_pr': 1e-6,
        'hessian_approximation': 'exact',  # 或者使用 'gauss-newton'
        'qpsol': 'qpoases',  # QP子问题求解器
        'qpsol_options': {
            'printLevel': 'none'
        }
    }
}

def solve_softobj_with_solver(start_coff, end_coff, solver_name: str):
    """使用指定求解器求解优化问题"""
    opti = ca.Opti()

    accept_pos = opti.variable(NTRAJ, NDIM)

    # 设置目标函数
    opti.minimize(create_obj_with_cons_fn(accept_pos, ca.DM(start_coff), ca.DM(end_coff)))
    # opti.set_initial(accept_pos, ca.DM(np.repeat(ca.DM(start_coff).full()[np.newaxis,0,:],NTRAJ, axis=0)))
    opti.solver(solver_name, solver_specific_options[solver_name])
    sol = opti.solve()
    return np.array(ca.DM(sol.value(accept_pos))).reshape(-1,2)



def evaluate(current_coff, accept_pos):
    '''  整条轨迹评估函数，输入每条traj时间，所有中间点，转角，计算评估点 '''
    traj_n = accept_pos.shape[0]

    positions = np.zeros((traj_n*NPIECE*NDRAW_PT, NDIM))
    velocities = np.zeros((traj_n*NPIECE*NDRAW_PT, NDIM))
    accelerates = np.zeros((traj_n*NPIECE*NDRAW_PT, NDIM))
    curvatures_sq = np.zeros((traj_n*NPIECE*NDRAW_PT, 1))
    jerks = np.zeros((traj_n*NPIECE*NDRAW_PT, NDIM))
    snaps = np.zeros((traj_n*NPIECE*NDRAW_PT, NDIM))

    poly_eval_fn = create_poly_eval_func(n_piece=NPIECE, n_drawpt=NDRAW_PT)

    for i in range(traj_n):
        new_coff = solve_coff(current_coff=current_coff, pieceT=PIECE_T, end_pos=accept_pos.reshape(-1,2)[i,:].reshape(1,-1))
        current_coff = new_coff

        # 本段轨迹的piece时长
        result = poly_eval_fn.call({"T":PIECE_T/RATIO, "coff":current_coff})


        start_ind = i*NPIECE*NDRAW_PT
        end_ind = start_ind + NPIECE*NDRAW_PT
        positions[start_ind:end_ind, :] = ca.DM(result["pos_ckpts"])
        velocities[start_ind:end_ind, :] = ca.DM(result["vel_ckpts"])
        accelerates[start_ind:end_ind, :] = ca.DM(result["acc_ckpts"])
        curvatures_sq[start_ind:end_ind, :] = ca.DM(result["curvature_sq_ckpts"])
        jerks[start_ind:end_ind, :] = ca.DM(result["jerk_ckpts"])
        snaps[start_ind:end_ind, :] = ca.DM(result["snap_ckpts"])



    eval_result = {
        "positions":positions,
        "velocities":velocities,
        "accelerates":accelerates,
        "curvatures_sq":curvatures_sq,
        "jerks":jerks,
        "snaps":snaps,
    }
    create_visualization(eval_result=eval_result, total_time=PIECE_T*RATIO*accept_pos.shape[0])


def main():
    # cost_fn = create_obj_with_cons_fn()

    start_state = np.array([[2.0,0.0], [-0.5,0], [0.0,0.0]])
    start_coff = ca.DM(state2coff(start_state))

    end_state = np.array([[10.0,5.0], [0.0,0], [0.0,0.0]])
    end_coff = ca.DM(state2coff(end_state))

    accept_pos = []
    current_coff = start_coff
    for i in range(5):
        # print(create_obj_with_cons_fn(np.array([[10,0]]), current_coff, end_coff))
        # print(create_obj_with_cons_fn(np.array([[9,0]]), current_coff, end_coff))
        # print(create_obj_with_cons_fn(np.array([[8,0]]), current_coff, end_coff))
        # print(create_obj_with_cons_fn(np.array([[3,0]]), current_coff, end_coff))
        # print(create_obj_with_cons_fn(np.array([[3,3]]), current_coff, end_coff))
        # ap = np.array([[10,0]])
        ap = solve_softobj_with_solver(start_coff=current_coff, end_coff=end_coff, solver_name='ipopt')

        current_coff = solve_coff(current_coff=current_coff, pieceT=PIECE_T, end_pos=ap[0,:].reshape(1,-1))

        accept_pos.append(ap[0])
        print(ap)

    evaluate(current_coff=state2coff(start_state), accept_pos=np.array(accept_pos))


if __name__ == '__main__':
    main()
    # test_obj_func()
