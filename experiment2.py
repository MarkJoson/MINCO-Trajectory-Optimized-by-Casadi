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


def constructMincoM(T):
    mat_m = np.zeros((NCOFF, NCOFF), dtype=np.float64)
    for i in range(NCOFF-1):
        mat_m[i, :] = constructBetaT(0, i).T

    mat_m[-1, :] = constructBetaT(T, 0).T
    return mat_m

def constructMincoQ(last_coff, tgtPos, T):
    mat_q = np.zeros((NCOFF, NDIM), dtype=np.float64)
    mat_ck = np.zeros((NCOFF-1, NCOFF), dtype=np.float64)

    mat_ck[0, :] = constructBetaT(T, 0).T
    mat_ck[1, :] = constructBetaT(T, 1).T
    mat_ck[2, :] = constructBetaT(T, 2).T
    mat_ck[3, :] = constructBetaT(T, 3).T
    mat_ck[4, :] = constructBetaT(T, 4).T

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
    print(mat_m)
    return new_coff

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
       [1,0]])

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
        tgt_pos = tgts[i,:] #np.random.rand(1,2)

        coff = iter_func(pieceT, tgt_pos, coff)   # 迭代更新coff

        # end_pt = tb.constructBetaT(pieceT, 0).T @ coff
        # start_pt = tb.constructBetaT(0, 0).T @ coff

        # end_vel = tb.constructBetaT(pieceT, 1).T @ coff
        # start_vel = tb.constructBetaT(0, 1).T @ coff
        # print(f"start={start_pt}, end={end_pt}")
        # print(f"Vs={start_vel}, Ve={end_vel}")

        coffs.append(coff)


    coffs_origin = np.array(
        [
 [ 5.7920000000004843e-02, -5.5919135999999323e+00],
 [ 1.1998125292923757e-02, -3.2599999999998602e-01],
 [ 7.0290532492519513e-04,  1.1248242462115489e-02],
 [ 1.5251638739698453e-06,  1.2203217446611515e-04],
 [-5.3610665777800087e-07, -5.7193645274001185e-06],
 [ 5.5835719187426311e-09,  4.4675554815197365e-08],
 [ 3.1744000000001166e-01, -7.8040991999998761e+00],
 [ 2.6995781909076609e-02, -2.0000000000008383e-02],
 [ 3.5145266246252210e-04,  1.1248242462115228e-02],
 [-1.6776802613689818e-05, -9.7625739572827474e-05],
 [-1.7870221925875957e-07, -2.8596822637086600e-06],
 [ 5.5835719187160025e-09,  4.4675554815162293e-08],
 [ 6.8256000000000772e-01, -6.4829327999999400e+00],
 [ 2.6995781909076727e-02,  2.0199999999999610e-01],
 [-3.5145266246257566e-04,  5.6241212310575994e-03],
 [-1.6776802613690936e-05, -1.7084504425270887e-04],
 [ 1.7870221925825403e-07,  8.8429340926480630e-18],
 [ 5.5835719187983393e-09,  4.4675554814517595e-08],
 [ 9.4208000000000169e-01, -3.3182783999999916e+00],
 [ 1.1998125292922150e-02,  2.6799999999999763e-01],
 [-7.0290532492509180e-04,  7.0967386846459443e-17],
 [ 1.5251638739744201e-06, -9.7625739572918452e-05],
 [ 5.3610665777711742e-07,  2.8596822636949216e-06],
 [ 5.5835719187743519e-09,  4.4675554814902029e-08],
        ]
    )


    coffs = np.array(coffs)
    coffs = coffs.reshape(-1, 2)
    print(coffs)
    print(coffs-coffs_origin)
    # coffs[-1,:] = coffs_origin[-1,:]
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
