# pylint: disable=C0103,C0111,C0301
import numpy as np
import casadi as ca
from config import *
from toolbox import *
from debug_casadi import *

def constructMincoM(T):
    mat_m = SYM_TYPE(NCOFF, NCOFF)
    for i in range(NCOFF-1):
        mat_m[i, :] = constructBetaT(0, i)

    mat_m[-1, :] = constructBetaT(T, 0)
    return mat_m

def constructMincoQ(start_coff, tgtPos, T):
    mat_q = SYM_TYPE(NCOFF, NDIM)
    mat_ck = SYM_TYPE(NCOFF-1, NCOFF)
    for i in range(NCOFF-1):
        mat_ck[i, :] = constructBetaT(T, i)
    mat_q[:-1, :] = mat_ck @ start_coff
    mat_q[-1, :] = tgtPos
    return mat_q

def constructInitialCoff(init_pos):
    coff = SYM_TYPE(NCOFF, NDIM)
    coff[0,:] = init_pos
    return coff

def buildIterFunc():
    pieceT = SYM_TYPE.sym('T')
    tgt_pos = SYM_TYPE.sym('tgt_pos', NDIM)
    init_coff = SYM_TYPE.sym('init_coff', NCOFF, NDIM)

    mat_m = constructMincoM(pieceT)
    mat_q = constructMincoQ(init_coff, tgt_pos, pieceT)
    new_coff = ca.solve(mat_m, mat_q)

    return ca.Function('iter_fn', [pieceT, tgt_pos, init_coff], [new_coff])


def main():
    iter_fn = buildIterFunc()
    pieceT = SYM_TYPE.sym('T')
    init_pos = np.array([0, 0])
    
    tgt_pos = np.array([3,5])
    init_coff = constructInitialCoff(init_pos)
    new_coff = iter_fn(pieceT, tgt_pos, init_coff)

    print_structured_matrix(init_coff)
    print_structured_matrix(new_coff)

if __name__ == '__main__':
    main()
