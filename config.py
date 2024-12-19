import casadi as ca

S         = 3         # jerk控制
POLY_RANK = 2*S-1     # 多项式次数
NCOFF     = 2*S       # 轨迹系数个数
NDIM      = 2         # 轨迹维数

SYM_TYPE = ca.MX
