# pylint: disable=C0103,C0111,C0301
import math
import numpy as np
import casadi as ca
from visualize import create_visualization
from debug_casadi import print_matrix, print_structured_matrix
from typing import Dict, List, Tuple
import time


NDIM      = 2         # 轨迹维数
NTRAJ     = 2         # 轨迹条数
NPIECE    = 2         # 5阶段轨迹曲线
NMIDPT    = NTRAJ*NPIECE-1      # 中间路径点（优化变量）个数

S         = 3         # jerk控制
POLY_RANK = 2*S-1     # 多项式次数
NCOFF     = 2*S       # 轨迹系数个数
NCKPT     = 10        # 检查点个数
NDRAW_PT  = 50

# 车辆换档时的速度
VEL_SHIFT = 0.5

# 可行性参数
max_vel_sq = 2
max_acc_sq = 3
max_cur_sq = 0.5

# 优化权重
weight_dt = 10
weight_vel = 100.0
weight_acc = 100.0
weight_cur = 100.0


def constructBetaT(t, rank:int):
    ''' 构造特定时间的β(t) '''
    beta = ca.MX(NCOFF, 1)
    for i in range(rank, NCOFF):
        if not isinstance(t, int|float) or t!=0 or i-rank==0:
            beta[i] = math.factorial(i)/math.factorial(i-rank) * t**(i-rank)
    return beta

def constructEi(T):
    ''' 构造M矩阵中的Ei(2s*2s)=[β(T), β(T), ..., β(T)^(2s-2)] '''
    Ei = ca.MX(NCOFF, NCOFF)
    Ei[0, :] = constructBetaT(T, 0)
    for i in range(1, NCOFF):
        Ei[i, :] = constructBetaT(T, i-1)
    return Ei

def constructFi():
    ''' 构造M矩阵中的Fi(2s*2s)=[0, -β(0), ..., β(0)^(2s-2)] '''
    Fi = ca.MX(NCOFF, NCOFF)
    for i in range(1, NCOFF):
        Fi[i, :] = -constructBetaT(0, i-1)
    return Fi

def constructF0():
    ''' 构造M矩阵中的F0(s*2s)=[β(0), ..., β(0)^(s-1)] '''
    F0 = ca.MX(S, NCOFF)      # 端点约束
    for i in range(S):
        F0[i, :] = constructBetaT(0, i)
    return F0

def constructEM(T):
    ''' 构造M矩阵中的E0(s*2s)=[β(T), ..., β(T)^(s-1)] '''
    E0 = ca.MX(S, NCOFF)      # 端点约束
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
    M = ca.MX(num_pieces*NCOFF, num_pieces*NCOFF)
    # F0 = ca.MX.sym('F0', 3, 6)
    # EM = ca.MX.sym('EM', 3, 6)
    M[0:S, 0:NCOFF] = constructF0()         # 3*6
    M[-S:, -NCOFF:] = constructEM(T=pieceT) # 3*6
    for i in range(1, num_pieces):
        # Fi = ca.MX.sym(f'F{i}', 6, 6)
        # Ei = ca.MX.sym(f'E{i}', 6, 6)
        M[(i-1)*NCOFF+S:i*NCOFF+S, (i-1)*NCOFF:i*NCOFF] = constructEi(pieceT)
        M[(i-1)*NCOFF+S:i*NCOFF+S, i*NCOFF:(i+1)*NCOFF] = constructFi()
    return M

def constructB(state0, stateT, mid_pos, num_pieces):
    ''' 构造右端路径点约束B矩阵'''
    B = ca.MX(num_pieces*NCOFF, NDIM)
    B[0:S,:] = state0                     # 起点状态
    B[-S:,:] = stateT                       # 终点状态
    for i in range(1, num_pieces):
        B[(i-1)*NCOFF+S, :] = mid_pos[i-1, :]      # 设置中间路径点
    return B

def constructBBTint(pieceT, rank):
    ''' c^T*(∫β*β^T*dt)*c ==> (2, NCOFF) * (NCOFF, NCOFF) * (NCOFF, 2) '''
    bbint = ca.MX(NCOFF, NCOFF)
    beta = constructBetaT(pieceT, rank)
    for i in range(NCOFF):
        for j in range(NCOFF):
            if i+j-2*rank < 0: continue
            coff = 1 / (i+j-2*rank+1)
            bbint[i, j] = coff * beta[i] * beta[j] * pieceT
    return bbint

def create_L1_function():
    a0 = 1e-3
    x = ca.MX.sym('x')

    f1 = 0  # x <= 0
    f2 = -1/(2*a0**3)*x**4 + 1/a0**2*x**3  # 0 < x <= a0
    f3 = x - a0/2  # a0 < x

    L1 = ca.if_else(x <= 0, f1,
         ca.if_else(x <= a0, f2, f3))

    L1_func = ca.Function('L1', [x], [L1])

    return L1_func

def create_T_function():
    tau = ca.MX.sym('tau')

    f1 = 0.5 * tau**2 + tau + 1  # tau > 0
    f2 = (tau**2 - 2*tau + 2)/2  # tau <= 0

    Tf = ca.if_else(tau > 0, f1, f2)

    T_func = ca.Function('T', [tau], [Tf])

    return T_func

def constructCkptMat(pieceT, num_ckpt:int, rank:int):
    ''' 构造单个piece的CKPT检查矩阵 '''
    ckpt_mat = ca.MX(NCOFF, num_ckpt)
    ckpt_frac = [(i+1)/(num_ckpt+1) for i in range(num_ckpt)]
    ckpt_ts = ckpt_frac * pieceT

    for i in range(num_ckpt):
        ckpt_mat[:, i] = constructBetaT(t=ckpt_ts[i], rank=rank)
    return ckpt_mat

def constructNPiecesCkptMat(pieceT, rank:int, nckpt:int, npiece:int):
    ''' 构造整条轨迹的ckpt检查矩阵[NPIECE*NCOFF, NPIECE*NCKPT] '''
    ckpt_mat = ca.MX(npiece*NCOFF, npiece*nckpt)
    for i in range(npiece):
        ckpt_mat[i*NCOFF:(i+1)*NCOFF, i*nckpt:(i+1)*nckpt] = constructCkptMat(pieceT=pieceT, num_ckpt=nckpt, rank=rank)
    return ckpt_mat
    # return ca.repmat(constructCkptMat(pieceT=pieceT, num_ckpt=nckpt, rank=rank), npiece, 1)

def constrainCostFunc(pieceT, coff_mat):

    L1slack = create_L1_function()

    vel_ckm = constructNPiecesCkptMat(pieceT=pieceT, rank=1, nckpt=NCKPT, npiece=NPIECE)
    acc_ckm = constructNPiecesCkptMat(pieceT=pieceT, rank=2, nckpt=NCKPT, npiece=NPIECE)

    # [NPIECE*NCOFF, NCKPT].T @ [NPIECE*NCOFF, NDIM] -> [NUM_CKPT, NDIM]
    vels = vel_ckm.T @ coff_mat
    accs = acc_ckm.T @ coff_mat

    vels_sqsum = ca.sum2(vels ** 2)
    accs_sqsum = ca.sum2(accs ** 2)

    # AUXB = ca.DM([[0,-1],[1,0]])
    # curvature_sq = ca.sum2((accs @ AUXB) * vels)**2 / vels_sqsum ** 3
    numerator = (vels[:,0]*accs[:,1] - vels[:,1]*accs[:,0])**2
    denominator = vels_sqsum**3
    curvature_sq = numerator / (denominator+0.001)

    con_vel = ca.sum1(L1slack(vels_sqsum - max_vel_sq))
    con_acc = ca.sum1(L1slack(accs_sqsum - max_acc_sq))
    con_cur = ca.sum1(L1slack(curvature_sq - max_cur_sq))

    return weight_vel*con_vel + weight_acc*con_acc + weight_cur*con_cur

def constructStateByPosAndDir(pos, vel_theta, sign):
    ret = ca.MX(S, NDIM)
    ret[0, :] = pos
    ret[1, 0] = ca.cos(vel_theta) * VEL_SHIFT * sign
    ret[1, 1] = ca.sin(vel_theta) * VEL_SHIFT * sign
    return ret


def objectFuncWithConstrain(mid_pos, vel_angles, traj_ts_free, start_state, end_state):
    ''' 代价函数 '''
    cost =  0
    Tfn = create_T_function()
    # 中间点的个数为: NTraj * NPiece - 1
    # 其中i*NPiece是轨迹i和轨迹i+1的交界
    # [i*NPiece, (i+1)*NPiece-1) 一共Npiece-1个点是中间点
    # 换挡点（切换点）: i*NPIECE-1
    # ×  ▲  ▲  ▲  ●  ▲  ▲  ▲  ●  ▲  ▲  ▲  ●  ▲  ▲  ▲  ×
    #    -  -  -  ↑  -  -  -  ↑  -  -  -  ↑  -  -  -
    #    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
    for i in range(NTRAJ):
        trajTs = Tfn(traj_ts_free[i])
        # pieceT = ca.MX.sym('T')
        pieceT = trajTs / NPIECE
        traj_mid_pos = mid_pos[i*NPIECE:(i+1)*NPIECE-1, :]

        if i==0:
            state0 = start_state
        else:
            state0 = constructStateByPosAndDir(mid_pos[i*NPIECE-1, :], vel_angles[i-1,0], -1)      # 速度与上一段相反

        if i==NTRAJ-1:
            stateT = end_state
        else:
            stateT = constructStateByPosAndDir(mid_pos[(i+1)*NPIECE-1, :], vel_angles[i,0], 1)

        M = constructM(pieceT=pieceT, num_pieces=NPIECE)
        bbint = constructBBTint(pieceT=pieceT, rank=S)      # (ncoff * ncoff)
        B = constructB(state0=state0, stateT=stateT, mid_pos=traj_mid_pos, num_pieces=NPIECE)
        c = ca.solve(M, B)


        for j in range(NPIECE):
            cj = c[j*NCOFF:(j+1)*NCOFF, :]
            cost += ca.trace(cj.T @ bbint @ cj)
        cost += pieceT * weight_dt

        cost += constrainCostFunc(pieceT=pieceT, coff_mat=c)
    return cost


def test_obj_func():
    mid_pos = ca.MX.sym('mid_pos', NTRAJ*NPIECE-1, NDIM)
    vel_angles = ca.MX.sym('vel_angles', NTRAJ-1)
    traj_ts_free = ca.MX.sym('traj_ts_free', NTRAJ)
    start_state = ca.MX.sym('start_state', S, NDIM)
    end_state = ca.MX.sym('end_state', S, NDIM)

    objectFuncWithConstrain(
        mid_pos=mid_pos,
        vel_angles=vel_angles,
        traj_ts_free=traj_ts_free,
        start_state=start_state,
        end_state=end_state)

def optimize_soft_constrain(start_state_np, end_state_np):
    opti = ca.Opti()

    start_state = ca.DM(start_state_np)
    end_state = ca.DM(end_state_np)
    # traj_ts_free = ca.DM([2,2])

    mid_pos = opti.variable(NMIDPT, NDIM)
    vel_angles = opti.variable(NTRAJ-1, 1)
    traj_ts_free = opti.variable(NTRAJ, 1)

    opti.set_initial(mid_pos, np.array([[0,1],[1,3],[1,1]]))
    opti.set_initial(vel_angles, np.array([ca.pi/2]))
    opti.set_initial(traj_ts_free, np.array([0,0]))

    opti.minimize(objectFuncWithConstrain(mid_pos=mid_pos, vel_angles=vel_angles, traj_ts_free=traj_ts_free, start_state=start_state, end_state=end_state))

    # opti.solver('bonmin')
    # SQP求解器配置
    sqp_opts = {
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

    # 设置求解器
    # opti.solver('sqpmethod', sqp_opts)
    opti.solver('ipopt')
    opti.solver('bonmin')

    sol = opti.solve()
    print(sol.value(mid_pos))
    print(sol.value(vel_angles))
    print(sol.value(traj_ts_free))

    eval_and_show(
        mid_pos_np=sol.value(mid_pos),
        vel_angles_np=sol.value(vel_angles),
        traj_ts_free_np=sol.value(traj_ts_free),
        start_state_np=start_state_np,
        end_state_np=end_state_np
    )


def compare_solvers(start_state_np, end_state_np):
    # 定义要测试的求解器列表
    solvers = ['ipopt']  # 可以根据实际安装情况调整
    results = {}

    for solver_name in solvers:
        try:
            result = solve_with_solver(
                start_state_np,
                end_state_np,
                solver_name
            )
            results[solver_name] = result
        except Exception as e:
            print(f"Solver {solver_name} failed: {str(e)}")
            results[solver_name] = None

    # 打印比较结果
    print_comparison(results)

    # 可视化最佳结果
    best_solver = find_best_solver(results)
    if best_solver:
        print(f"\nBest solver: {best_solver}")
        best_result = results[best_solver]
        eval_and_show(
            mid_pos_np=best_result['mid_pos'],
            vel_angles_np=best_result['vel_angles'],
            traj_ts_free_np=best_result['traj_ts_free'],
            start_state_np=start_state_np,
            end_state_np=end_state_np
        )

    return results

def solve_with_solver(start_state_np, end_state_np, solver_name: str) -> Dict:
    """使用指定求解器求解优化问题"""
    opti = ca.Opti()

    start_state = ca.DM(start_state_np)
    end_state = ca.DM(end_state_np)

    # 定义变量
    mid_pos = opti.variable(NMIDPT, NDIM)
    vel_angles = opti.variable(NTRAJ-1, 1)
    traj_ts_free = opti.variable(NTRAJ, 1)

    # 设置初值
    opti.set_initial(mid_pos, np.array([[0,1],[1,3],[1,1],[1,3],[1,1]]))
    opti.set_initial(vel_angles, np.array([ca.pi/2,ca.pi/2]))
    opti.set_initial(traj_ts_free, np.array([0,0,0]))

    # 设置目标函数
    opti.minimize(objectFuncWithConstrain(
        mid_pos=mid_pos,
        vel_angles=vel_angles,
        traj_ts_free=traj_ts_free,
        start_state=start_state,
        end_state=end_state
    ))

    # 配置求解器选项
    solver_options = get_solver_options(solver_name)
    opti.solver(solver_name, solver_options)

    # 计时并求解
    start_time = time.time()
    try:
        sol = opti.solve()
        solve_time = time.time() - start_time

        # 计算目标函数值
        obj_value = sol.value(opti.f)

        return {
            'success': True,
            'solve_time': solve_time,
            'objective_value': obj_value,
            'mid_pos': sol.value(mid_pos),
            'vel_angles': sol.value(vel_angles),
            'traj_ts_free': sol.value(traj_ts_free),
            'solver_stats': opti.stats()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'solve_time': time.time() - start_time
        }

def get_solver_options(solver_name: str) -> Dict:
    """返回求解器特定的选项"""
    common_options = {
        'print_time': True,
        'ipopt.print_level': 0,
        'ipopt.sb': 'yes'
    }

    solver_specific_options = {
        'bonmin': {
            # 'bonmin.algorithm': 'B-BB',
            # 'bonmin.solution_limit': 1
        },
        'ipopt': {
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0,  # 0-12
            'print_time': True,
            'ipopt.tol': 1e-6,      # 收敛容差
            'ipopt.hessian_approximation': 'limited-memory',  # 使用L-BFGS

        },
        'knitro': {
            'knitro.algorithm': 1,
            'knitro.mir_maxiter': 1000
        }
    }

    options = common_options.copy()
    if solver_name in solver_specific_options:
        options.update(solver_specific_options[solver_name])

    return options

def print_comparison(results: Dict):
    """打印求解器比较结果"""
    print("\nSolver Comparison Results:")
    print("-" * 80)
    print(f"{'Solver':<15} {'Status':<10} {'Time (s)':<12} {'Objective':<15} {'Iterations':<10}")
    print("-" * 80)

    for solver_name, result in results.items():
        if result and result['success']:
            status = "Success"
            solve_time = f"{result['solve_time']:.4f}"
            obj_value = f"{result['objective_value']:.4e}"
            iterations = result['solver_stats']['iter_count'] if 'iter_count' in result['solver_stats'] else 'N/A'
        else:
            status = "Failed"
            solve_time = "N/A"
            obj_value = "N/A"
            iterations = "N/A"

        print(f"{solver_name:<15} {status:<10} {solve_time:<12} {obj_value:<15} {iterations:<10}")

    print("-" * 80)

def find_best_solver(results: Dict) -> str:
    """根据目标函数值和求解时间找到最佳求解器"""
    valid_results = {
        name: result for name, result in results.items()
        if result and result['success']
    }

    if not valid_results:
        return None

    # 这里可以根据需要调整选择标准
    # 当前使用目标函数值作为主要标准
    return min(
        valid_results.items(),
        key=lambda x: x[1]['objective_value']
    )[0]


def createEvalFunc():
    # mid_pos, vel_dirs, trajTs, start_state, end_state
    Tfn = create_T_function()

    mid_pos = ca.MX.sym('mid_pos', NTRAJ*NPIECE-1, NDIM)
    vel_angles = ca.MX.sym('vel_angles', NTRAJ-1)
    traj_ts_free = ca.MX.sym('traj_ts_free', NTRAJ)
    start_state = ca.MX.sym('start_state', S, NDIM)
    end_state = ca.MX.sym('end_state', S, NDIM)

    positions = ca.MX(NTRAJ*NPIECE*NDRAW_PT, NDIM)
    velocities = ca.MX(NTRAJ*NPIECE*NDRAW_PT, NDIM)
    accelerates = ca.MX(NTRAJ*NPIECE*NDRAW_PT, NDIM)
    curvatures_sq = ca.MX(NTRAJ*NPIECE*NDRAW_PT, 1)

    for i in range(NTRAJ):
        trajTs = Tfn(traj_ts_free[i])
        pieceT = trajTs / NPIECE
        traj_mid_pos = mid_pos[i*NPIECE:(i+1)*NPIECE-1, :]

        if i==0:
            state0 = start_state
        else:
            state0 = constructStateByPosAndDir(mid_pos[i*NPIECE-1, :], vel_angles[i-1], -1)      # 速度与上一段相反

        if i==NTRAJ-1:
            stateT = end_state
        else:
            stateT = constructStateByPosAndDir(mid_pos[(i+1)*NPIECE-1, :], vel_angles[i], 1)

        M = constructM(pieceT=pieceT, num_pieces=NPIECE)
        B = constructB(state0=state0, stateT=stateT, mid_pos=traj_mid_pos, num_pieces=NPIECE)
        c = ca.solve(M, B)

        pos_ckpt_mat = constructNPiecesCkptMat(pieceT=pieceT, rank=0, nckpt=NDRAW_PT, npiece=NPIECE)
        vel_ckpt_mat = constructNPiecesCkptMat(pieceT=pieceT, rank=1, nckpt=NDRAW_PT, npiece=NPIECE)
        acc_ckpt_mat = constructNPiecesCkptMat(pieceT=pieceT, rank=2, nckpt=NDRAW_PT, npiece=NPIECE)

        pos_ckpts = pos_ckpt_mat.T @ c
        vel_ckpts = vel_ckpt_mat.T @ c
        acc_ckpts = acc_ckpt_mat.T @ c

        vels_sqsum = ca.sum2(vel_ckpts ** 2)
        numerator = (vel_ckpts[:,0]*acc_ckpts[:,1] - vel_ckpts[:,1]*acc_ckpts[:,0])**2
        denominator = vels_sqsum**3
        curvature_sq_ckpts = numerator / (denominator+0.001)

        start_ind = i*NPIECE*NDRAW_PT
        end_ind = start_ind + NPIECE*NDRAW_PT
        positions[start_ind:end_ind, :] = pos_ckpts
        velocities[start_ind:end_ind, :] = vel_ckpts
        accelerates[start_ind:end_ind, :] = acc_ckpts
        curvatures_sq[start_ind:end_ind, :] = curvature_sq_ckpts

    return ca.Function(
        'eval_traj',
        [mid_pos, vel_angles, traj_ts_free, start_state, end_state],
        [positions, velocities, accelerates, curvatures_sq],
        ['mid_pos', 'vel_angles', 'traj_ts_free', 'start_state', 'end_state'],
        ['positions', 'velocities', 'accelerates', 'curvatures_sq']
        )

def eval_and_show(mid_pos_np, vel_angles_np, traj_ts_free_np, start_state_np, end_state_np):
    eval_traj = createEvalFunc()
    Tfn = create_T_function()

    mid_pos = ca.DM(mid_pos_np)
    vel_angles = ca.DM(vel_angles_np)  # 竖直向上
    traj_ts_free = ca.DM(traj_ts_free_np)         # 每段2秒
    start_state = ca.DM(start_state_np)
    end_state = ca.DM(end_state_np)
    eval_result = eval_traj.call(
        {
            'mid_pos': mid_pos,
            'vel_angles': vel_angles,
            'traj_ts_free': traj_ts_free,
            'start_state': start_state,
            'end_state': end_state
        }
    )

    # print(eval_result)
    create_visualization(eval_result=eval_result, total_time=np.sum(Tfn(traj_ts_free_np)))

def evaluate():
    mid_pos = [
        [0, 1],
        [1, 1],
        [1, 0],
    ]
    # mid_pos = np.random.rand(NMIDPT, 2)*10
    vel_angles = np.random.rand(NTRAJ-1)
    traj_ts_free = np.random.rand(NTRAJ)

    # vel_angles = [1.0406699880175019]
    # traj_ts_free = [-6.24084667 -5.29853158]
    start_state = [
        [0.0,0.0],
        [0.0,0.25],
        [0.0,0.0]
    ]
    end_state = [
        [0.0,0.0],
        [0.0,0.25],
        [0.0,0.0]
    ]
    print(mid_pos)
    eval_and_show(mid_pos, vel_angles, traj_ts_free, start_state, end_state)


def main():
    start_state = ca.DM([
        [0.0,0.0],
        [0.0,0.25],
        [0.0,0.0]
    ])
    end_state = ca.DM([
        [1.0,3.0],
        [0.0,0.25],
        [0.0,0.0]
    ])

    # test_eval()
    # compare_solvers(start_state, end_state)
    optimize_soft_constrain(start_state, end_state)

if __name__ == '__main__':
    main()
    # test_obj_func()


