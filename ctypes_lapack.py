import numpy, time
import scipy.linalg

A = scipy.linalg.toeplitz(numpy.r_[-2, 1, numpy.zeros(1000)]/1000.)

print 'SciPy eigh'
t0 = time.time()
e1 = scipy.linalg.eigvalsh(A).real
print time.time() - t0

##########################################


LAPACK_LIB = '/home/miro3/.hashdist/bld/openblas/c7qemuijpr3x/lib/libopenblas.so'
#LAPACK_LIB = '/home/miro3/Documents/Programming/julia/usr/lib/libopenblas64_.so'

 
def eigh_ctypes(A):
    import ctypes
    from ctypes import c_int, c_char, c_void_p, c_voidp
    # The routine overwrites the argument, so lets make a copy
    A2 = numpy.array(A, copy=True, order='C', dtype=float)
    N, M = A2.shape
    assert N == M
    
    # Some helpers
    LAPACK_ROW_MAJOR = 101
    array1d_t = numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=1)
    array2d_t = numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=2)
    
    # Define the function
    lapack = ctypes.CDLL(LAPACK_LIB)
    dgeev = lapack.LAPACKE_dgeev
    # JOBVL, JOBVR, N, A, LDA,
    # WR, WI, VL, LDVL, VR, LDVR
    # WORK, LWORK, INFO
    dgeev.restype = c_int
    dgeev.argtypes = [c_int, c_char, c_char,
                      c_int, array2d_t, c_int,
                      array1d_t, array1d_t,
                      array1d_t, c_int, array1d_t, c_int]
    
    LDA = N
    WR = numpy.zeros(N, float)
    WI = numpy.zeros(N, float)
    LDVL = LDVR = N
    VL = numpy.zeros(LDVL, float)
    VR = numpy.zeros(LDVR, float)
    info = dgeev(LAPACK_ROW_MAJOR, 'N', 'N',
                 N, A2, LDA,
                 WR, WI,
                 VL, LDVL, VR, LDVR)
    assert info == 0, "Got non-zero return value %d from LAPACK" % info
    return WR

print 'CTypes eigh'
t0 = time.time()
e2 = eigh_ctypes(A)
print time.time() - t0

import numpy as np
print np.linalg.norm(e1-e2, np.inf)
