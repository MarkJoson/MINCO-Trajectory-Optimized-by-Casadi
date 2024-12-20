# pylint: disable=C0103,C0111,C0301 W0718,W0401
import time
from typing import Dict, List, Tuple

import numpy as np
import casadi as ca
from evaluate import create_visualization
from debug_casadi import print_matrix, print_structured_matrix

from config import *
from toolbox import *

np.set_printoptions(precision=50)

# 轨迹优化参数
NTRAJ     = 1         # 轨迹条数
NPIECE    = 5         # 5阶段轨迹曲线
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
weight_dt = 10
weight_vel = 0*1000.0
weight_acc = 0*1000.0
weight_cur = 0*1000.0


def create_state_by_pos_dir_func():
    pos = ca.SX.sym('pos', NDIM)
    vel_amp = ca.SX.sym('vel_amp')
    vel_theta = ca.SX.sym('vel_theta')
    vel_sign = ca.SX.sym('vel_sign')

    ret = ca.SX(S, NDIM)
    ret[0, :] = pos
    ret[1, 0] = ca.cos(vel_theta) * vel_amp * vel_sign
    ret[1, 1] = ca.sin(vel_theta) * vel_amp * vel_sign
    return ca.Function('state_by_pos_dir', [pos, vel_amp, vel_theta, vel_sign], [ret])


def objectFuncWithConstrain(mid_pos, vel_angles, traj_ts_free, start_state, end_state):
    ''' 代价函数
    中间点的个数为: NTraj * NPiece - 1
    其中i*NPiece是轨迹i和轨迹i+1的交界
    [i*NPiece, (i+1)*NPiece-1) 一共Npiece-1个点是中间点
    换挡点（切换点）: i*NPIECE-1
    ×  ▲  ▲  ▲  ●  ▲  ▲  ▲  ●  ▲  ▲  ▲  ●  ▲  ▲  ▲  ×
       -  -  -  ↑  -  -  -  ↑  -  -  -  ↑  -  -  -
       0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 '''

    cost =  0
    traj_ts = tau2T_func(traj_ts_free)
    state_by_pos_dir_fn = create_state_by_pos_dir_func()
    for i in range(NTRAJ):
        # 本段轨迹的piece时长
        pieceT = traj_ts[i] / NPIECE

        # 计算本段轨迹的起点和终点
        state0 = start_state if i==0 else state_by_pos_dir_fn(mid_pos[i*NPIECE-1, :], VEL_SHIFT, vel_angles[i-1,0], -1)
        stateT = end_state if i==NTRAJ-1 else state_by_pos_dir_fn(mid_pos[(i+1)*NPIECE-1, :], VEL_SHIFT, vel_angles[i,0], 1)

        # 求解本段轨迹的多项式系数矩阵
        traj_mid_pos = mid_pos[i*NPIECE:(i+1)*NPIECE-1, :]
        M = constructM(pieceT=pieceT, num_pieces=NPIECE)
        bbint = constructBBTint(pieceT=pieceT, rank=S)      # (ncoff * ncoff)
        B = constructB(state0=state0, stateT=stateT, mid_pos=traj_mid_pos, num_pieces=NPIECE)
        c = ca.solve(M, B)

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

        vels_sqsum = ca.sum2(vels ** 2)
        accs_sqsum = ca.sum2(accs ** 2)

        numerator = (vels[:,0]*accs[:,1] - vels[:,1]*accs[:,0])**2
        denominator = vels_sqsum**3
        curvature_sq = numerator / (denominator)#+1e-5)

        con_vel = ca.sum1(L1_func(vels_sqsum - max_vel_sq))
        con_acc = ca.sum1(L1_func(accs_sqsum - max_acc_sq))
        con_cur = ca.sum1(L1_func(curvature_sq - max_cur_sq))

        cost += pieceT * weight_dt + weight_vel*con_vel + weight_acc*con_acc + weight_cur*con_cur
    return cost

def test_obj_func():
    ''' 使用代数变量带入参数方便检查错误 '''
    mid_pos = SYM_TYPE.sym('mid_pos', NTRAJ*NPIECE-1, NDIM) # type: ignore
    vel_angles = SYM_TYPE.sym('vel_angles', NTRAJ-1) # type: ignore
    traj_ts_free = SYM_TYPE.sym('traj_ts_free', NTRAJ) # type: ignore
    start_state = SYM_TYPE.sym('start_state', S, NDIM) # type: ignore
    end_state = SYM_TYPE.sym('end_state', S, NDIM) # type: ignore

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
    traj_ts_free = ca.DM(np.ones((NTRAJ))*8)
    # traj_ts_free = opti.variable(NTRAJ, 1)

    # 设置初值

    # opti.set_initial(mid_pos, np.array([[-0.00264075,  0.11095673],
    #    [-0.01577599,  0.17299636],
    #    [ 0.5       ,  0.05792456],
    #    [ 1.01577598,  0.17299636],
    #    [ 1.00264075,  0.11095673]]))
    opti.set_initial(mid_pos, np.array([end_state_np[0,:]*(i+1)/(NMIDPT+1)+start_state_np[0,:]*(1-(i+1)/(NMIDPT+1)) for i in range(NMIDPT)]))
    opti.set_initial(vel_angles, np.array([ca.pi]))
    # opti.set_initial(traj_ts_free, np.array([0.1]))

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

    return results


def evaluate(mid_pos, vel_angles, traj_ts_free, start_state, end_state):
    '''  整条轨迹评估函数，输入每条traj时间，所有中间点，转角，计算评估点 '''
    positions = np.zeros((NTRAJ*NPIECE*NDRAW_PT, NDIM))
    velocities = np.zeros((NTRAJ*NPIECE*NDRAW_PT, NDIM))
    accelerates = np.zeros((NTRAJ*NPIECE*NDRAW_PT, NDIM))
    curvatures_sq = np.zeros((NTRAJ*NPIECE*NDRAW_PT, 1))
    jerks = np.zeros((NTRAJ*NPIECE*NDRAW_PT, NDIM))
    snaps = np.zeros((NTRAJ*NPIECE*NDRAW_PT, NDIM))

    mid_pos = np.array(mid_pos).reshape(-1, NDIM)
    vel_angles = np.array(vel_angles).reshape(-1)
    traj_ts_free = np.array(traj_ts_free).reshape(-1)

    traj_eval_fn = create_traj_eval_func(n_piece=NPIECE, n_drawpt=NDRAW_PT, n_mid_pos=NPIECE-1)
    state_by_pos_dir_fn = create_state_by_pos_dir_func()

    traj_ts = tau2T_func(traj_ts_free)

    for i in range(NTRAJ):
        pieceT = traj_ts[i] / NPIECE
        traj_mid_pos = mid_pos[i*NPIECE:(i+1)*NPIECE-1, :]

        state0 = start_state if i==0 else state_by_pos_dir_fn(mid_pos[i*NPIECE-1, :], VEL_SHIFT, vel_angles[i-1], -1)
        stateT = end_state if i==NTRAJ-1 else state_by_pos_dir_fn(mid_pos[(i+1)*NPIECE-1, :], VEL_SHIFT, vel_angles[i], 1)

        # stateT = np.array(stateT.to_DM())
        result = traj_eval_fn.call({"T":pieceT, "state0":state0, "stateT":stateT, "mid_pos":traj_mid_pos})

        start_ind = i*NPIECE*NDRAW_PT
        end_ind = start_ind + NPIECE*NDRAW_PT
        positions[start_ind:end_ind, :] = result["pos_ckpts"]
        velocities[start_ind:end_ind, :] = result["vel_ckpts"]
        accelerates[start_ind:end_ind, :] = result["acc_ckpts"]
        curvatures_sq[start_ind:end_ind, :] = result["curvature_sq_ckpts"]
        jerks[start_ind:end_ind, :] = result["jerk_ckpts"]
        snaps[start_ind:end_ind, :] = result["snap_ckpts"]

        print(np.array(result["coff"]))

        # print(f"----------{constructBetaT(0,0,ca.SX).T @ result['coff'][:6,:]}")
        # print(f"----------{constructBetaT(0,1,ca.SX).T @ result['coff'][:6,:]}")
        # print(f"----------{constructBetaT(0,2,ca.SX).T @ result['coff'][:6,:]}")
        # print(f"----------{constructBetaT(0,3,ca.SX).T @ result['coff'][:6,:]}")
        # print(f"----------{constructBetaT(0,4,ca.SX).T @ result['coff'][:6,:]}")


    eval_result = {
        "positions":positions,
        "velocities":velocities,
        "accelerates":accelerates,
        "curvatures_sq":curvatures_sq,
        "jerks":jerks,
        "snaps":snaps,
    }
    create_visualization(eval_result=eval_result, total_time=np.sum(traj_ts))


def main():
    start_state = np.array([
        [0.0,0.0],
        [0.0,-0.5],
        [0.0,0.0]
    ])
    end_state = np.array([
        [1.0,0.0],
        [0.0,0.25],
        [0.0,0.0]
    ])

    result = solve_softobj_with_solver(start_state_np=start_state, end_state_np=end_state, solver_name='ipopt')
    print(result)

    evaluate(
        mid_pos=result['mid_pos'],
        vel_angles=result['vel_angles'],
        traj_ts_free=result['traj_ts_free'],
        start_state=start_state,
        end_state=end_state
    )


if __name__ == '__main__':
    main()
    # test_obj_func()
