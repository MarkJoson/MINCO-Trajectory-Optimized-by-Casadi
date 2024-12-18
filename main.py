# pylint: disable=C0103,C0111,C0301
import math
import numpy as np
import casadi as ca

import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec


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
max_cur_sq = 5

# 优化权重
weight_dt = 0.1
weight_vel = 0#10.0
weight_acc = 0#10.0
weight_cur = 0#10.0


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
    curvature_sq = numerator / denominator

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
        bbint = constructBBTint(pieceT=pieceT, rank=S)      # (ncoff * ncoff)
        B = constructB(state0=state0, stateT=stateT, mid_pos=traj_mid_pos, num_pieces=NPIECE)
        c = ca.solve(M, B)


        for j in range(NPIECE):
            cj = c[j*NCOFF:(j+1)*NCOFF]
            cost += ca.trace(cj.T @ bbint @ cj)
        cost += traj_ts_free[i] * weight_dt

        cost += constrainCostFunc(pieceT=pieceT, coff_mat=c)
    return cost

# 1. 基本的打印辅助函数
def print_matrix(matrix, name="Matrix"):
    """更易读的矩阵打印函数"""
    print(f"\n{name}:")
    if isinstance(matrix, ca.MX):
        rows, cols = matrix.shape
        for i in range(rows):
            row_str = "["
            for j in range(cols):
                elem = matrix[i,j]
                # 尝试简化表达式
                if elem.is_constant():
                    row_str += f"{float(elem):8.3f}"
                else:
                    row_str += f"{str(elem):8s}"
                row_str += " "
            row_str += "]"
            print(row_str)

# 2. 带有详细信息的矩阵分析
def analyze_matrix(matrix, name="Matrix"):
    """详细分析矩阵结构"""
    print(f"\n=== Analysis of {name} ===")
    print(f"Shape: {matrix.shape}")
    print(f"Number of elements: {matrix.numel()}")
    print(f"Symbolic variables: {[str(v) for v in ca.symvar(matrix)]}")

    # 打印每个元素的详细信息
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            elem = matrix[i,j]
            print(f"\nElement [{i},{j}]:")
            print(f"  Expression: {str(elem)}")
            print(f"  Is symbolic: {elem.is_symbolic()}")
            print(f"  Is constant: {elem.is_constant()}")

# 4. 结构化展示复杂矩阵
def print_structured_matrix(matrix):
    """结构化展示矩阵内容"""
    rows, cols = matrix.shape

    # 获取元素的最大字符长度
    max_length = 0
    for i in range(rows):
        for j in range(cols):
            max_length = max(max_length, len(str(matrix[i,j])))

    # 打印矩阵
    print("\nMatrix structure:")
    print("─" * (cols * (max_length + 3) + 1))
    for i in range(rows):
        print("│", end=" ")
        for j in range(cols):
            elem = str(matrix[i,j])
            print(f"{elem:{max_length}}", end=" │ ")
        print("\n" + "─" * (cols * (max_length + 3) + 1))

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

        start_ind = i*NPIECE*NDRAW_PT
        end_ind = start_ind + NPIECE*NDRAW_PT
        positions[start_ind:end_ind, :] = pos_ckpt_mat.T @ c
        velocities[start_ind:end_ind, :] = vel_ckpt_mat.T @ c
        accelerates[start_ind:end_ind, :] = acc_ckpt_mat.T @ c

    return ca.Function(
        'eval_traj',
        [mid_pos, vel_angles, traj_ts_free, start_state, end_state],
        [positions, velocities, accelerates],
        ['mid_pos', 'vel_angles', 'traj_ts_free', 'start_state', 'end_state'],
        ['positions', 'velocities', 'accelerates']
        )


def create_visualization(eval_result, total_time):
    """
    Create an animated visualization of the trajectory optimization.

    Args:
        eval_result: Dictionary containing 'positions', 'velocities', 'accelerates' from eval_traj
        total_time: Total time duration for the trajectory
    """
    # Convert casadi DM objects to numpy arrays and reshape
    pos = np.array(eval_result['positions'])
    vel = np.array(eval_result['velocities'])
    acc = np.array(eval_result['accelerates'])

    # Reshape arrays from (n,2) to (2,n)
    n_points = pos.shape[0]
    pos = pos.reshape(n_points, 2).T
    vel = vel.reshape(n_points, 2).T
    acc = acc.reshape(n_points, 2).T

    # Create time vector
    time = np.linspace(0, total_time, n_points)

    # Calculate derived quantities
    speed_magnitude = np.sum(vel**2, axis=0)
    accel_magnitude = np.sum(acc**2, axis=0)

    # Calculate curvature
    B = np.array([[0, -1], [1, 0]])
    curvature = np.zeros(n_points)
    for i in range(n_points):
        v = vel[:, i]
        a = acc[:, i]
        v_sq_sum = np.sum(v**2)
        if v_sq_sum > 1e-6:
            curvature[i] = np.sum((B @ a) * v)**2 / (v_sq_sum**3 + 1e-6)

    # Create figure and subplots
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, figure=fig)

    # Trajectory subplot
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.grid(True)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('X Position')
    ax_traj.set_ylabel('Y Position')
    ax_traj.set_title('Motion Trajectory')

    # Set axis limits with some padding
    x_min, x_max = pos[0].min(), pos[0].max()
    y_min, y_max = pos[1].min(), pos[1].max()
    padding = 0.2 * max(x_max - x_min, y_max - y_min)
    ax_traj.set_xlim(x_min - padding, x_max + padding)
    ax_traj.set_ylim(y_min - padding, y_max + padding)

    # Plot full trajectory
    ax_traj.plot(pos[0, :], pos[1, :], 'b-', alpha=0.3, label='Trajectory')

    # Create moving objects
    particle, = ax_traj.plot([], [], 'bo', markersize=10, label='Particle')
    vel_arrow = ax_traj.quiver([], [], [], [], color='r', scale=20, label='Velocity')
    acc_arrow = ax_traj.quiver([], [], [], [], color='g', scale=20, label='Acceleration')
    ax_traj.legend()

    # Speed subplot
    ax_speed = fig.add_subplot(gs[0, 1])
    ax_speed.grid(True)
    ax_speed.set_ylabel('Speed Magnitude')
    ax_speed.set_title('Speed vs Time')
    ax_speed.plot(time, speed_magnitude, 'b-', alpha=0.3)
    speed_point, = ax_speed.plot([], [], 'r.', markersize=10)
    speed_line = ax_speed.axvline(x=time[0], color='k', linestyle='--')

    # Acceleration subplot
    ax_accel = fig.add_subplot(gs[1, 1])
    ax_accel.grid(True)
    ax_accel.set_ylabel('Acceleration Magnitude')
    ax_accel.plot(time, accel_magnitude, 'b-', alpha=0.3)
    accel_point, = ax_accel.plot([], [], 'g.', markersize=10)
    accel_line = ax_accel.axvline(x=time[0], color='k', linestyle='--')

    # Curvature subplot
    ax_curv = fig.add_subplot(gs[2, 1])
    ax_curv.grid(True)
    ax_curv.set_xlabel('Time (s)')
    ax_curv.set_ylabel('Curvature')
    ax_curv.plot(time, curvature, 'b-', alpha=0.3)
    curv_point, = ax_curv.plot([], [], 'g.', markersize=10)
    curv_line = ax_curv.axvline(x=time[0], color='k', linestyle='--')

    def init():
        particle.set_data([], [])
        speed_point.set_data([], [])
        accel_point.set_data([], [])
        curv_point.set_data([], [])
        return particle, speed_point, accel_point, curv_point

    def animate(i):
        # Update particle position - 修正这里
        particle.set_data([pos[0, i]], [pos[1, i]])  # 使用列表包装单个值

        # Update velocity arrow
        vel_arrow.set_offsets(np.column_stack((pos[0, i], pos[1, i])))
        vel_arrow.set_UVC(vel[0, i], vel[1, i])

        # Update acceleration arrow
        acc_arrow.set_offsets(np.column_stack((pos[0, i], pos[1, i])))
        acc_arrow.set_UVC(acc[0, i], acc[1, i])

        # Update speed plot
        speed_point.set_data(time[:i+1], speed_magnitude[:i+1])
        speed_line.set_xdata([time[i], time[i]])

        # Update acceleration plot
        accel_point.set_data(time[:i+1], accel_magnitude[:i+1])
        accel_line.set_xdata([time[i], time[i]])

        # Update curvature plot
        curv_point.set_data(time[:i+1], curvature[:i+1])
        curv_line.set_xdata([time[i], time[i]])

        return particle, vel_arrow, acc_arrow, speed_point, accel_point, curv_point

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_points,
                        interval=50, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()

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

def test_eval():
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


def test_opt_soft_constrain(start_state_np, end_state_np):
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

    opti.solver('ipopt')


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


# def getConstraints(pieceT, coff_mat):
#     """Returns inequality constraints for velocity, acceleration and curvature"""
#     vel_ckm = constructNPiecesCkptMat(pieceT=pieceT, rank=1, nckpt=NCKPT, npiece=NPIECE)
#     acc_ckm = constructNPiecesCkptMat(pieceT=pieceT, rank=2, nckpt=NCKPT, npiece=NPIECE)

#     # 计算每个检查点的速度和加速度
#     vels = vel_ckm.T @ coff_mat
#     accs = acc_ckm.T @ coff_mat

#     # 计算各个物理量
#     vels_sqsum = ca.sum2(vels ** 2)
#     accs_sqsum = ca.sum2(accs ** 2)

#     # 计算曲率
#     numerator = (vels[:,0]*accs[:,1] - vels[:,1]*accs[:,0])**2
#     denominator = vels_sqsum**3 + 1e-3  # 添加小量避免除零

#     # 对每个检查点返回约束
#     return {
#         'g_vel': vels_sqsum,      # <= max_vel_sq
#         'g_acc': accs_sqsum,      # <= max_acc_sq
#         'g_cur': numerator / denominator  # <= max_cur_sq
#     }

# def test_opt(start_state_np, end_state_np):
#     opti = ca.Opti()

#     start_state = ca.DM(start_state_np)
#     end_state = ca.DM(end_state_np)
#     traj_ts_free = opti.variable(NTRAJ, 1)  # 现在时间也作为优化变量

#     mid_pos = opti.variable(NMIDPT, NDIM)
#     vel_angles = opti.variable(NTRAJ-1, 1)

#     # 设置初值
#     # opti.set_initial(mid_pos, np.array([[0,1],[2,2],[2,2],[2,2],[3,1]]))
#     # opti.set_initial(vel_angles, np.array([ca.pi/2]))
#     # opti.set_initial(traj_ts_free, np.array([2, 2]))

#     # 时间约束
#     opti.subject_to(traj_ts_free >= -15)  # 每段最小时间
#     opti.subject_to(traj_ts_free <= 15.0)  # 每段最大时间

#     # 计算目标函数
#     cost = 0
#     Tfn = create_T_function()

#     # 对每段轨迹添加约束
#     for i in range(NTRAJ):
#         trajTs = Tfn(traj_ts_free[i])
#         pieceT = trajTs / NPIECE

#         # 获取当前段的中间点
#         traj_mid_pos = mid_pos[i*NPIECE:(i+1)*NPIECE-1, :]

#         # 确定起点和终点状态
#         if i==0:
#             state0 = start_state
#         else:
#             state0 = constructStateByPosAndDir(mid_pos[i*NPIECE-1], vel_angles[i-1], -1)

#         if i==NTRAJ-1:
#             stateT = end_state
#         else:
#             stateT = constructStateByPosAndDir(mid_pos[(i+1)*NPIECE-1], vel_angles[i], 1)

#         # 构造并求解轨迹系数
#         M = constructM(pieceT=pieceT, num_pieces=NPIECE)
#         bbint = constructBBTint(pieceT=pieceT, rank=S)
#         B = constructB(state0=state0, stateT=stateT, mid_pos=traj_mid_pos, num_pieces=NPIECE)
#         c = ca.solve(M, B)

#         # 添加代价函数
#         for j in range(NPIECE):
#             cj = c[j*NCOFF:(j+1)*NCOFF]
#             cost += ca.trace(cj.T @ bbint @ cj)
#         cost += weight_dt * trajTs

#         # 添加约束
#         constraints = getConstraints(pieceT, c)
#         opti.subject_to(constraints['g_vel'] <= max_vel_sq)
#         opti.subject_to(constraints['g_acc'] <= max_acc_sq)
#         opti.subject_to(constraints['g_cur'] <= max_cur_sq)

#     # 设置目标函数
#     opti.minimize(cost)

#     # 求解器配置
#     opts = {
#         'ipopt.print_level': 3,
#         'ipopt.tol': 1e-4,
#         'ipopt.max_iter': 3000,
#         'ipopt.warm_start_init_point': 'yes',
#         'ipopt.mu_strategy': 'adaptive'
#     }
#     opti.solver('ipopt', opts)

#     # try:
#     sol = opti.solve()
#     print("Optimization succeeded!")
#     print("Mid positions:", sol.value(mid_pos))
#     print("Velocity angles:", sol.value(vel_angles))
#     print("Trajectory times:", sol.value(traj_ts_free))

#     eval_and_show(
#         mid_pos_np=sol.value(mid_pos),
#         vel_angles_np=sol.value(vel_angles),
#         traj_ts_free_np=sol.value(traj_ts_free),
#         start_state_np=start_state_np,
#         end_state_np=end_state_np
#     )
#     # except:
#     #     print("Optimization failed!")
#     #     print(opti.debug.value(mid_pos))
#     #     print(opti.debug.value(vel_angles))
#     #     print(opti.debug.value(traj_ts_free))

def main():
    start_state = ca.DM([
        [0.0,0.0],
        [0.0,0.25],
        [0.0,0.0]
    ])
    end_state = ca.DM([
        [2.0,0.0],
        [0.0,-0.25],
        [0.0,0.0]
    ])

    # test_eval()
    test_opt_soft_constrain(start_state, end_state)
    # test_opt(start_state, end_state)

if __name__ == '__main__':
    main()


