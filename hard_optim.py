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
