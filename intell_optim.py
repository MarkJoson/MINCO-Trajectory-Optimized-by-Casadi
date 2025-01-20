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
