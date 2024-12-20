# pylint: disable=C0103,C0111,C0301
import math
import numpy as np
from config import *
import toolbox as tb
from debug_casadi import *
from evaluate import create_visualization
import copy

ITER_TIMES = 4
DRAWPT_PER_PIECE = 50

def constructBetaT(t, rank:int):
    beta = np.zeros((NCOFF, 1), dtype=np.float64)
    for i in range(rank, NCOFF):
        if not isinstance(t, int|float) or t!=0 or i-rank==0:
            beta[i,0] = math.factorial(i)/math.factorial(i-rank) * t**(i-rank)
    return beta

def constructMincoM(pieceT):
    mat_m = np.zeros((NCOFF, NCOFF), dtype=np.float64)
    for i in range(NCOFF-1):
        mat_m[i, :] = constructBetaT(0, i).T
    mat_m[-1, :] = constructBetaT(pieceT, 0).T
    return mat_m









def main():
    eval_fn = tb.create_poly_eval_func(n_piece=ITER_TIMES, n_drawpt=DRAWPT_PER_PIECE)

    pieceT = (8**2 + 0.01) / 5
    init_pos = np.array([0, 0])

    # 初始化系数
    coff = np.zeros((NCOFF, NDIM))
    coff[0,:] = init_pos

    # 目标点序列
    tgts = np.array([
        [ 0.31744000000001155 , -7.804099199999875   ],
        [ 0.6825600000000079  , -6.482932799999943   ],
        [ 0.9420800000000021  , -3.318278399999993   ],
        [1,0]
    ])

    coffs = []
    for i in range(ITER_TIMES):
        tgt_pos = tgts[i,:]

        # 计算约束边界
        # bound_vel = bound_all(pieceT=pieceT, rank=1, val=0.4, coff=coff, nckpt=5)
        # bound_acc = bound_all(pieceT=pieceT, rank=2, val=0.03, coff=coff, nckpt=5)
        # bounds = np.concatenate([[bound_vel], [bound_acc]])
        # lb = np.max(bounds[:,0], axis=0)
        # rb = np.min(bounds[:,1], axis=0)
        # bound = np.array([lb, rb])

        # 对目标位置进行裁剪
        # new_tgt_pos = np.clip(tgt_pos, bound[0,:], bound[1,:])
        # print(f"iter:{i}, bound={bound}")
        # print(f"iter:{i}, past:{tgt_pos}, new:{new_tgt_pos}")

        # 使用新的迭代函数更新系数
        coff = iter_func_new(pieceT, tgt_pos, coff)
        coffs.append(coff)

    # 评估结果
    coffs = np.array(coffs)
    coffs = coffs.reshape(-1, 2)
    result = eval_fn.call({"T":pieceT, "coff":coffs})

    eval_result = {
        "positions":result["pos_ckpts"],
        "velocities": result["vel_ckpts"],
        "accelerates": result["acc_ckpts"],
        "curvatures_sq": result["curvature_sq_ckpts"],
        "jerks": result["jerk_ckpts"],
        "snaps": result["snap_ckpts"],
    }
    create_visualization(eval_result=eval_result, total_time=pieceT * ITER_TIMES)

if __name__ == '__main__':
    main()