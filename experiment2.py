# pylint: disable=C0103,C0111,C0301
import math
import numpy as np
from config import *
import toolbox as tb
from debug_casadi import *
from evaluate import create_visualization
import scipy.linalg as sl
import control

ITER_TIMES = 200
DRAWPT_PER_PIECE = 2

RATIO = 0.1


np.set_printoptions(2)

def constructBetaT(t, rank:int):
    ''' 构造特定时间的β(t) '''
    beta = np.zeros((NCOFF, 1), dtype=np.float64)
    for i in range(rank, NCOFF):
        if not isinstance(t, int|float) or t!=0 or i-rank==0:
            beta[i,0] = math.factorial(i)/math.factorial(i-rank) * t**(i-rank)
    return beta

def constructBBTint(pieceT, rank):
    ''' c^T*(∫β*β^T*dt)*c ==> (2, NCOFF) * (NCOFF, NCOFF) * (NCOFF, 2) '''
    bbint = np.zeros((NCOFF, NCOFF))
    beta = constructBetaT(pieceT, rank)
    for i in range(NCOFF):
        for j in range(NCOFF):
            if i+j-2*rank < 0: continue
            coff = 1 / (i+j-2*rank+1)
            bbint[i, j] = coff * beta[i,0] * beta[j,0] * pieceT
    return bbint


def constructMincoM(pieceT):
    mat_m = np.zeros((NCOFF, NCOFF), dtype=np.float64)
    for i in range(NCOFF-1):
        mat_m[i, :] = constructBetaT(0, i).T

    mat_m[-1, :] = constructBetaT(pieceT, 0).T
    return mat_m

def constructMincoM2(pieceT):

    mat_m = np.zeros((NCOFF, NCOFF), dtype=np.float64)
    for i in range(NCOFF-2):
        mat_m[i, :] = constructBetaT(0, i).T

    mat_m[-2, :] = constructBetaT(pieceT, 0).T
    mat_m[-1, :] = constructBetaT(pieceT, 1).T

    mat_m_inv = np.linalg.inv(mat_m)
    mat_supp = np.array([[0,0,0,0,0,1]]) @ mat_m_inv @ constructBBTint(pieceT=pieceT, rank=S)

    mat_m[-1, :] = mat_supp[-1, :]

    return mat_m



def constructMincoQ(last_coff, tgtPos, pieceT):
    mat_r = consturctMatR(pieceT=pieceT)
    mat_q = mat_r @ last_coff
    mat_q[-1, :] = tgtPos
    return mat_q

def consturctMatR(pieceT):
    mat_r = np.zeros((NCOFF, NCOFF), dtype=np.float64)
    for i in range(NCOFF):
        mat_r[i, :] = constructBetaT(pieceT, i).T
    return mat_r

def constructInitialCoff(init_pos):
    coff = np.zeros((NCOFF, NDIM), dtype=np.float64)
    coff[0,:] = init_pos
    return coff
    # init_state = ca.SX.sym('init_state', NCOFF, NDIM) # type: ignore

def calc_target_coefficient(u):
    """计算目标系数c*(u)"""
    c_star = np.zeros((NCOFF, NDIM))
    c_star[0,:] = u  # 常数项设为目标位置
    return c_star

def calc_lyapunov(c):
    """计算李亚普诺夫函数V(c)"""
    # ts = [((i+1)/(nckpt+1)*pieceT) for i in range(nckpt)]

    Q = np.diag([0, 10,100,1000,0,0])
    V = np.sum(c.T @ Q @ c)  # V = eᵀQe
    return V

def nonlinear_scaling(c, c_star, k=0.01):
    """非线性缩放函数α(θ)"""

    curV = calc_lyapunov(c)
    dV = calc_lyapunov(c_star) - curV
    maxV = calc_lyapunov(np.array([[0,0], [0.1,0.1], [0.1,0.1], [0,0], [0,0], [0,0],]))

    return 1#1 / (1.0 + np.exp((curV-maxV)/k))

def compute_control_input(coff_now, c_star, c_stable, kappa=0.1):
    """计算非线性控制输入f(u,c)"""
    alpha = nonlinear_scaling(coff_now, c_star)
    error = c_star - coff_now
    f = alpha * (c_stable-c_star) + c_star

    return f

def finite_horizon_lqr(A, B, Q, R, QN, N):
    # 获取系统维度
    n = A.shape[0]  # 状态维度
    m = B.shape[1]  # 输入维度

    # 初始化序列
    P_sequence = [None] * (N + 1)
    K_sequence = [None] * N

    # 终端条件
    P_sequence[N] = QN

    # 向后递推
    for t in range(N-1, -1, -1):
        # 计算Riccati方程
        P_next = P_sequence[t + 1]

        # 计算K(t)
        K = -np.linalg.inv(R + B.T @ P_next @ B) @ B.T @ P_next @ A
        K_sequence[t] = K

        # 更新P(t)
        P = Q + A.T @ P_next @ A + A.T @ P_next @ B @ K
        P_sequence[t] = P

    return K_sequence, P_sequence

class Planner:
    def __init__(self, pieceT) -> None:
        self.pieceT = pieceT / RATIO
        mat_m = constructMincoM2(self.pieceT)
        mat_m_inv = np.linalg.inv(mat_m)
        mat_r = consturctMatR(pieceT=pieceT)
        mat_s = np.diag([1,1,1,1,0,0])
        mat_u = np.array([[0,0,0,0,1,0]]).T

        # mat_h = constructBetaT(pieceT, 0).T

        self.mat_F = mat_m_inv @ mat_s @ mat_r              # matF是行向量
        self.mat_G = (mat_m_inv @ mat_u).reshape(-1,1)      # matG是列向量

        # LQR
        Q = constructBBTint(pieceT=pieceT, rank=3)# + np.diag([1,1,1,0,0,0])*3
        # Q = np.diag([1,0,0,0,0,0])*1
        # Q[0,0] = 1e2
        R = np.array([[1]])     # 控制权重矩阵
        self.K, _, _ = control.dlqr(self.mat_F, self.mat_G, Q, R)
        # Kseq,Pseq = finite_horizon_lqr(self.mat_F, self.mat_G, Q, R, np.diag([1,1,1,1,1,1])*10, 30)
        # self.K = Kseq[0]
        self.Kpp = sl.pinv(self.mat_G) @ (np.identity(6)-self.mat_F)+self.K

        # STAB
        self.mat_F_stab = self.mat_F - self.mat_G @ self.K
        self.mat_G_stab = self.mat_G @ self.Kpp @ np.array([[1,0,0,0,0,0]]).T

    def iter_func(self, tgt_pos, last_coff):
        new_coff = self.mat_F @ last_coff + self.mat_G @ tgt_pos.reshape(1,-1)
        return new_coff

    def iter_func_new(self, tgt_pos, last_coff):
        """修改后的迭代更新函数"""
        tgt_pos = tgt_pos.reshape(1,-1)

        new_coff = self.mat_F_stab@last_coff + self.mat_G_stab@tgt_pos

        return new_coff

    # pT = (beta^T * M^-1 * b) * [vi - beta^T * M^-1 * S * c_old]
    def calc_bound(self, t, rank, val, coff):
        const_coeff = constructBetaT(t, rank).T @ self.mat_F_stab
        bound_coeff = constructBetaT(t, rank).T @ self.mat_G_stab

        val = -val if bound_coeff < 0 else val

        return np.array([(-val - const_coeff@coff)/bound_coeff, (val - const_coeff@coff)/bound_coeff])


    def bound_all(self, coff, nckpt):
        ts = [((i+1)/(nckpt+1)*self.pieceT) for i in range(nckpt)]
        vel_bounds = np.concatenate([self.calc_bound(t, 1, 2, coff) for t in ts], axis=1)
        acc_bounds = np.concatenate([self.calc_bound(t, 2, 3, coff) for t in ts], axis=1)
        # jerk_bounds = np.concatenate([self.calc_bound(t, 3, 300, coff) for t in ts], axis=1)
        bounds = np.concatenate([vel_bounds, acc_bounds], axis=1)

        lb = np.max(bounds[0,:], axis=0)
        ub = np.min(bounds[1,:], axis=0)
        return np.array([lb, ub])


def main():
    # iter_fn = buildIterFunc()
    eval_fn = tb.create_poly_eval_func(n_piece=ITER_TIMES, n_drawpt=DRAWPT_PER_PIECE)

    # pieceT = (8**2 + 0.01) / 5
    pieceT = 0.1
    init_pos = np.array([0, 0])
    tgt_pos = np.array([1, 0])

    planner = Planner(pieceT=pieceT)

    square_tgts = np.array([
       [0,10],
       [10,10],
       [10,0],
       [0,0],
    ])

    coffs = list()
    coff = constructInitialCoff(init_pos)

    for i in range(ITER_TIMES):
        tgt_pos = square_tgts[(i//40)%4,:]
        # tgt_pos = np.random.rand(2)
        # tgt_pos = np.array([20,20])

        # [upper[x, y], lower[x, y]]
        bound = planner.bound_all(coff=coff, nckpt=100)
        new_tgt_pos = np.clip(tgt_pos, bound[0], bound[1])

        # print("bound.x=(%.2f, %.2f), y(%.2f,%.2f) ---- old_tgt=(%.2f,%.2f) ---- new_tgt=(%.2f,%.2f)" % (
        #     bound[0][0],bound[1][0], bound[0][0],bound[1][0],
        #     tgt_pos[0],tgt_pos[1],new_tgt_pos[0],new_tgt_pos[1],))
        # print("\n\n")

        # coff = planner.iter_func(new_tgt_pos, coff)
        coff = planner.iter_func_new(new_tgt_pos, coff)

        # print(constructBetaT(pieceT, 0).T @ coff)
        coffs.append(coff)

    coffs = np.array(coffs)
    coffs = coffs.reshape(-1, 2)
    result = eval_fn.call({"T":pieceT , "coff":coffs})


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
