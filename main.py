# pylint: disable=C0103,C0111,C0301
import time
from typing import Dict, List, Tuple

import numpy as np
import casadi as ca
from visualize import create_visualization
from debug_casadi import print_matrix, print_structured_matrix

from config import *
from toolbox import *

# 轨迹优化参数
NTRAJ     = 2         # 轨迹条数
NPIECE    = 2         # 5阶段轨迹曲线
NMIDPT    = NTRAJ*NPIECE-1      # 中间路径点（优化变量）个数
NCKPT     = 20        # 检查点个数
NDRAW_PT  = 50

# 车辆换档时的速度
VEL_SHIFT = 0.025

# 可行性参数
max_vel_sq = 2
max_acc_sq = 3
max_cur_sq = 1.5

# 优化权重
weight_dt = 0.1
weight_vel = 1000.0
weight_acc = 1000.0
weight_cur = 1000.0

def constructStateByPosAndDir(pos, vel_theta, sign):
    ret = SYM_TYPE(S, NDIM)
    ret[0, :] = pos
    ret[1, 0] = ca.cos(vel_theta) * VEL_SHIFT * sign
    ret[1, 1] = ca.sin(vel_theta) * VEL_SHIFT * sign
    return ret

def objectFuncWithConstrain(mid_pos, vel_angles, traj_ts_free, start_state, end_state):
    ''' 代价函数 '''
    cost =  0
    # Tfn = create_softmax_function()
    Tfn = create_T_function()
    L1slack = create_L1_function()
    # L1slack = create_softmax_function()
    # 中间点的个数为: NTraj * NPiece - 1
    # 其中i*NPiece是轨迹i和轨迹i+1的交界
    # [i*NPiece, (i+1)*NPiece-1) 一共Npiece-1个点是中间点
    # 换挡点（切换点）: i*NPIECE-1
    # ×  ▲  ▲  ▲  ●  ▲  ▲  ▲  ●  ▲  ▲  ▲  ●  ▲  ▲  ▲  ×
    #    -  -  -  ↑  -  -  -  ↑  -  -  -  ↑  -  -  -
    #    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
    for i in range(NTRAJ):
        trajTs = Tfn(traj_ts_free[i])
        # pieceT = SYM_TYPE.sym('T')
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

        # c = SYM_TYPE.sym('c',12,2)

        # Minimal Control Effort
        for j in range(NPIECE):
            cj = c[j*NCOFF:(j+1)*NCOFF, :]
            cost += ca.trace(cj.T @ bbint @ cj)

        # Soft-Constrain
        vel_ckm = constructNPiecesCkptMat(pieceT=pieceT, rank=1, nckpt=NCKPT, npiece=NPIECE)
        acc_ckm = constructNPiecesCkptMat(pieceT=pieceT, rank=2, nckpt=NCKPT, npiece=NPIECE)

        # [NPIECE*NCOFF, NCKPT].T @ [NPIECE*NCOFF, NDIM] -> [NUM_CKPT, NDIM]
        vels = vel_ckm.T @ c
        accs = acc_ckm.T @ c

        # vels = SYM_TYPE.sym('vels',4,2)
        # accs = SYM_TYPE.sym('accs',4,2)

        vels_sqsum = ca.sum2(vels ** 2)
        accs_sqsum = ca.sum2(accs ** 2)

        numerator = (vels[:,0]*accs[:,1] - vels[:,1]*accs[:,0])**2
        denominator = vels_sqsum**3
        curvature_sq = numerator / (denominator)#+1e-5)

        con_vel = ca.sum1(L1slack(vels_sqsum - max_vel_sq))
        con_acc = ca.sum1(L1slack(accs_sqsum - max_acc_sq))
        con_cur = ca.sum1(L1slack(curvature_sq - max_cur_sq))

        cost += pieceT * weight_dt + weight_vel*con_vel + weight_acc*con_acc + weight_cur*con_cur
    return cost

def test_obj_func():
    ''' 使用代数变量带入参数方便检查错误 '''
    mid_pos = SYM_TYPE.sym('mid_pos', NTRAJ*NPIECE-1, NDIM)
    vel_angles = SYM_TYPE.sym('vel_angles', NTRAJ-1)
    traj_ts_free = SYM_TYPE.sym('traj_ts_free', NTRAJ)
    start_state = SYM_TYPE.sym('start_state', S, NDIM)
    end_state = SYM_TYPE.sym('end_state', S, NDIM)

    objectFuncWithConstrain(
        mid_pos=mid_pos,
        vel_angles=vel_angles,
        traj_ts_free=traj_ts_free,
        start_state=start_state,
        end_state=end_state)

## ^-------------------------------求解----------------------------------####

def get_solver_options(solver_name: str) -> Dict:
    """返回求解器特定的选项"""
    common_options = {
        'print_time': True,
        # 'ipopt.print_level': 0,
        # 'ipopt.sb': 'yes'
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
            'ipopt.tol': 1e-2,      # 收敛容差
            'ipopt.acceptable_tol': 1e-2,
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

    options = common_options.copy()
    if solver_name in solver_specific_options:
        options.update(solver_specific_options[solver_name])

    return options


def solve_softobj_with_solver(start_state_np, end_state_np, solver_name: str) -> Dict:
    """使用指定求解器求解优化问题"""
    opti = ca.Opti()

    start_state = ca.DM(start_state_np)
    end_state = ca.DM(end_state_np)

    # 定义变量
    mid_pos = opti.variable(NMIDPT, NDIM)
    vel_angles = opti.variable(NTRAJ-1, 1)
    traj_ts_free = opti.variable(NTRAJ, 1)

    # 设置初值

    # opti.set_initial(mid_pos, np.array([[-0.00264075,  0.11095673],
    #    [-0.01577599,  0.17299636],
    #    [ 0.5       ,  0.05792456],
    #    [ 1.01577598,  0.17299636],
    #    [ 1.00264075,  0.11095673]]))
    opti.set_initial(mid_pos, np.array([end_state_np[0,:]*(i+1)/(NMIDPT+1)+start_state_np[0,:]*(1-(i+1)/(NMIDPT+1)) for i in range(NMIDPT)]))
    opti.set_initial(vel_angles, np.array([ca.pi]))
    opti.set_initial(traj_ts_free, np.array([0.1]))

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
    opti.solver(solver_name)#, solver_options)

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

def compare_solvers(start_state_np, end_state_np):
    # 定义要测试的求解器列表
    solvers = ['ipopt']  # 可以根据实际安装情况调整
    results = {}

    for solver_name in solvers:
        try:
            result = solve_softobj_with_solver(
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

    ##### 根据目标函数值和求解时间找到最佳求解器 #####
    valid_results = {
        name: result for name, result in results.items()
        if result and result['success']
    }

    if not valid_results:
        return None

    best_solver = min(
        valid_results.items(),
        key=lambda x: x[1]['objective_value']       # 当前使用目标函数值作为主要标准
    )[0]

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


def createEvalFunc():
    # mid_pos, vel_dirs, trajTs, start_state, end_state
    Tfn = create_T_function()

    mid_pos = SYM_TYPE.sym('mid_pos', NTRAJ*NPIECE-1, NDIM)
    vel_angles = SYM_TYPE.sym('vel_angles', NTRAJ-1)
    traj_ts_free = SYM_TYPE.sym('traj_ts_free', NTRAJ)
    start_state = SYM_TYPE.sym('start_state', S, NDIM)
    end_state = SYM_TYPE.sym('end_state', S, NDIM)

    positions = SYM_TYPE(NTRAJ*NPIECE*NDRAW_PT, NDIM)
    velocities = SYM_TYPE(NTRAJ*NPIECE*NDRAW_PT, NDIM)
    accelerates = SYM_TYPE(NTRAJ*NPIECE*NDRAW_PT, NDIM)
    curvatures_sq = SYM_TYPE(NTRAJ*NPIECE*NDRAW_PT, 1)

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
        curvature_sq_ckpts = numerator / (denominator)

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
    start_state = np.array([
        [0.0,0.0],
        [0.0,0.25],
        [0.0,0.0]
    ])
    end_state = np.array([
        [1.0,0.0],
        [0.0,0.25],
        [0.0,0.0]
    ])

    result = solve_softobj_with_solver(start_state_np=start_state, end_state_np=end_state, solver_name='ipopt')
    print(result)

    eval_and_show(
        mid_pos_np=result['mid_pos'],
        vel_angles_np=result['vel_angles'],
        traj_ts_free_np=result['traj_ts_free'],
        start_state_np=start_state,
        end_state_np=end_state
    )



from scipy.optimize import differential_evolution, dual_annealing, direct

def create_objective_function():
    """创建可求值的目标函数"""
    # 创建符号变量
    mid_pos = SYM_TYPE.sym('mid_pos', NMIDPT, NDIM)
    vel_angles = SYM_TYPE.sym('vel_angles', NTRAJ-1, 1)
    traj_ts_free = SYM_TYPE.sym('traj_ts_free', NTRAJ, 1)
    start_state = SYM_TYPE.sym('start_state', S, NDIM)
    end_state = SYM_TYPE.sym('end_state', S, NDIM)

    # 构建目标函数
    cost = objectFuncWithConstrain(
        mid_pos=mid_pos,
        vel_angles=vel_angles,
        traj_ts_free=traj_ts_free,
        start_state=start_state,
        end_state=end_state
    )

    # 创建可调用的函数
    return ca.Function('objective',
                      [mid_pos, vel_angles, traj_ts_free, start_state, end_state],
                      [cost])

def evaluate_trajectory(x, obj_func, start_state_np, end_state_np):
    """将优化变量转换为目标函数值"""
    # 解包优化变量
    n_mid_pos = NMIDPT * NDIM
    n_vel_angles = NTRAJ - 1
    n_traj_ts = NTRAJ

    mid_pos = x[:n_mid_pos].reshape(NMIDPT, NDIM)
    vel_angles = x[n_mid_pos:n_mid_pos + n_vel_angles].reshape(NTRAJ-1, 1)
    traj_ts_free = x[n_mid_pos + n_vel_angles:].reshape(NTRAJ, 1)

    # 使用CasADi函数计算目标函数值
    cost = obj_func(mid_pos, vel_angles, traj_ts_free, start_state_np, end_state_np)

    return float(cost)

def solve_with_metaheuristic(start_state_np, end_state_np, method='de'):
    """使用元启发式算法求解"""
    # 创建目标函数
    obj_func = create_objective_function()

    # 确定优化变量的维度和范围
    n_mid_pos = NMIDPT * NDIM
    n_vel_angles = NTRAJ - 1
    n_traj_ts = NTRAJ
    total_vars = n_mid_pos + n_vel_angles + n_traj_ts

    # 设定变量范围
    bounds_pos = [(-5, 5)] * n_mid_pos  # 位置范围
    bounds_angle = [(-np.pi, np.pi)] * n_vel_angles  # 角度范围
    bounds_time = [(0.1, 10)] * n_traj_ts  # 时间范围
    bounds = bounds_pos + bounds_angle + bounds_time

    # 优化目标函数
    objective = lambda x: evaluate_trajectory(x, obj_func, start_state_np, end_state_np)

    start_time = time.time()

    if method == 'de':
        # 差分进化算法
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            updating='deferred',
            workers=1  # 使用单线程避免序列化问题
        )
    elif method == 'annealing':
        # 模拟退火算法
        result = dual_annealing(
            objective,
            bounds,
            maxiter=1000,
            initial_temp=5230.0,
            restart_temp_ratio=2e-5,
            visit=2.62,
            accept=-5.0
        )
    elif method == 'direct':
        # DIRECT算法
        result = direct(
            objective,
            bounds,
            maxiter=100,
            locally_biased=True
        )

    solve_time = time.time() - start_time

    if result.success:
        # 解包结果
        x_sol = result.x
        mid_pos = x_sol[:n_mid_pos].reshape(NMIDPT, NDIM)
        vel_angles = x_sol[n_mid_pos:n_mid_pos + n_vel_angles].reshape(NTRAJ-1, 1)
        traj_ts_free = x_sol[n_mid_pos + n_vel_angles:].reshape(NTRAJ, 1)

        return {
            'success': True,
            'solve_time': solve_time,
            'objective_value': result.fun,
            'mid_pos': mid_pos,
            'vel_angles': vel_angles,
            'traj_ts_free': traj_ts_free,
            'solver_stats': {
                'nfev': result.nfev,
                'nit': getattr(result, 'nit', None)
            }
        }
    else:
        return {
            'success': False,
            'error': 'Optimization failed',
            'solve_time': solve_time
        }

def compare_metaheuristic_solvers(start_state_np, end_state_np):
    """比较不同的元启发式算法"""
    methods = ['direct']
    results = {}

    for method in methods:
        try:
            print(f"\nTrying {method}...")
            result = solve_with_metaheuristic(
                start_state_np,
                end_state_np,
                method
            )
            results[method] = result
        except Exception as e:
            print(f"Method {method} failed: {str(e)}")
            results[method] = None

    print_comparison(results)
    return results

def main2():
    start_state = np.array([
        [0.0,0.0],
        [0.0,0.25],
        [0.0,0.0]
    ])
    end_state = np.array([
        [13.0,1.0],
        [0.0,0.25],
        [0.0,0.0]
    ])

    results = compare_metaheuristic_solvers(start_state, end_state)

    # 找到最佳结果并可视化
    valid_results = {k: v for k, v in results.items() if v and v['success']}
    if valid_results:
        best_method = min(valid_results.items(), key=lambda x: x[1]['objective_value'])[0]
        best_result = valid_results[best_method]
        print(f"\nBest method: {best_method}")

        eval_and_show(
            mid_pos_np=best_result['mid_pos'],
            vel_angles_np=best_result['vel_angles'],
            traj_ts_free_np=best_result['traj_ts_free'],
            start_state_np=start_state,
            end_state_np=end_state
        )


if __name__ == '__main__':
    main()
    # test_obj_func()


