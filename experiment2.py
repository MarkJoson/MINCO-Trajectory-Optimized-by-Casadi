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
    ''' 构造特定时间的β(t) '''
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

def constructMincoQ(last_coff, tgtPos, pieceT):
    mat_q = np.zeros((NCOFF, NDIM), dtype=np.float64)
    mat_ck = np.zeros((NCOFF-1, NCOFF), dtype=np.float64)

    for i in range(NCOFF-1):
        mat_ck[i, :] = constructBetaT(pieceT, i).T

    mat_q[:NCOFF-1, :] = mat_ck @ last_coff
    mat_q[-1, :] = tgtPos
    return mat_q

def constructInitialCoff(init_pos):
    coff = np.zeros((NCOFF, NDIM), dtype=np.float64)
    coff[0,:] = init_pos
    return coff
    # init_state = ca.SX.sym('init_state', NCOFF, NDIM) # type: ignore

def iter_func(pieceT, tgt_pos, last_coff):
    mat_m = constructMincoM(pieceT)
    mat_q = constructMincoQ(last_coff, tgt_pos, pieceT)
    new_coff = np.linalg.inv(mat_m) @ mat_q
    # new_coff = ca.solve(mat_m, mat_q)
    return new_coff




# pT = (beta^T * M^-1 * b) * [vi - beta^T * M^-1 * S * c_old]
def calc_bound(t, pieceT, rank, val, coff):
    mat_m = constructMincoM(pieceT)
    mat_m_inv = np.linalg.inv(mat_m)

    mat_r = np.zeros((NCOFF, NCOFF), dtype=np.float64)
    for i in range(NCOFF):
        mat_r[i, :] = constructBetaT(pieceT, i).T


    mat_s = np.diag([1,1,1,1,1,0])
    mat_u = np.array([[0,0,0,0,0,1]]).T

    tmp = constructBetaT(t, rank).T @ mat_m_inv

    mat_p = tmp @ mat_u
    mat_q = tmp @ mat_s @ mat_r

    if mat_p < 0:
        val = -val

    return np.array([(-val - mat_q@coff)/mat_p, (val - mat_q@coff)/mat_p])


def bound_all(pieceT, rank, val, coff, nckpt):
    ts = [((i+1)/(nckpt+1)*pieceT) for i in range(nckpt)]
    bounds = np.vstack([[calc_bound(t, pieceT, rank, val, coff)] for t in ts])

    lb = np.max(bounds[:,0], axis=0)
    rb = np.min(bounds[:,1], axis=0)
    return np.array([lb, rb])

# def buildIterFunc():
#     pieceT = ca.SX.sym('T') # type: ignore
#     tgt_pos = ca.SX.sym('tgt_pos', NDIM) # type: ignore
#     last_coff = ca.SX.sym('last_coff', NCOFF, NDIM) # type: ignore

#     mat_m = constructMincoM(pieceT)
#     mat_q = constructMincoQ(last_coff, tgt_pos, pieceT)
#     new_coff = ca.solve(mat_m, mat_q)

#     return ca.Function('iter_fn', [pieceT, tgt_pos, last_coff], [new_coff])


def main():
    # iter_fn = buildIterFunc()
    eval_fn = tb.create_poly_eval_func(n_piece=ITER_TIMES, n_drawpt=DRAWPT_PER_PIECE)

    pieceT = (8**2 + 0.01) / 5
    init_pos = np.array([0, 0])
    tgt_pos = np.array([1, 0])

    tgts = np.array([
        # [ 0.05791999999991071 , -0.03581759999628159 ],
       [ 0.31744000000001155 , -7.804099199999875   ],
       [ 0.6825600000000079  , -6.482932799999943   ],
       [ 0.9420800000000021  , -3.318278399999993   ],
       [1,0]
    ])

    coffs = list()
    # coff = constructInitialCoff(init_pos)
    coff = np.array([
        [-1.2631629528194653e-17,  1.5383701491068513e-15],
        [ 6.9025828012988348e-19, -5.0000000000000022e-01],
        [-2.3903564548352504e-18, -3.3124187071162540e-16],
        [ 3.8129096849303771e-05,  4.8812869786482533e-04],
        [-8.9351109629695727e-07, -8.5790467910970034e-06],
        [ 5.5835719187474023e-09,  4.4675554814892354e-08],
    ])

    for i in range(ITER_TIMES):
        tgt_pos = tgts[i,:]
        tgt_pos = np.random.rand(1,2)

        # [upper[x, y], lower[x, y]]
        bound_vel = bound_all(pieceT=pieceT, rank=1, val=0.4, coff=coff, nckpt=5)
        bound_acc = bound_all(pieceT=pieceT, rank=2, val=0.03, coff=coff, nckpt=5)
        bounds = np.concatenate([[bound_vel], [bound_acc]])
        lb = np.max(bounds[:,0], axis=0)
        rb = np.min(bounds[:,1], axis=0)
        bound = np.array([lb, rb])

        (rb-lb)

        new_tgt_pos = np.clip(tgt_pos, bound[0,0], bound[1,0])
        print(f"iter:{i}, bound={bound}")
        print(f"iter:{i}, past:{tgt_pos}, new:{new_tgt_pos}")

        coff = iter_func(pieceT, new_tgt_pos, coff)   # 迭代更新coff
        coffs.append(coff)

    coffs = np.array(coffs)
    coffs = coffs.reshape(-1, 2)
    # print(coffs)
    result = eval_fn.call({"T":pieceT, "coff":coffs})

    # "pos_ckpts", "vel_ckpts", "acc_ckpts", "curvature_sq_ckpts", "jerk_ckpts", "snap_ckpts"
    # positions = np.clip(result["pos_ckpts"], -5, 5)
    # print(positions)

    eval_result = {
        "positions":result["pos_ckpts"],
        "velocities": result["vel_ckpts"],
        "accelerates": result["acc_ckpts"],
        "curvatures_sq": result["curvature_sq_ckpts"],
        "jerks": result["jerk_ckpts"],
        "snaps": result["snap_ckpts"],
    }
    create_visualization(eval_result=eval_result, total_time=pieceT * ITER_TIMES)
    # print(pos_ckpts, vel_ckpts, acc_ckpts, curvature_sq_ckpts)



if __name__ == '__main__':
    main()
