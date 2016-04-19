import numpy as np

LAPACK_LIB = '/home/miro3/Documents/Programming/julia/usr/lib/libopenblas64_.so'

 
def s3d_eig(d, u):
    '''Eigen factorization of symmetric tridiagonal matrix.'''
    import ctypes
    from ctypes import c_int, c_char, c_double

    assert len(d) == len(u)+1
    # The routine overwrites the argument, so lets make a copy
    D = np.array(d, copy=True, order='F', dtype=float)
    E = np.array(u, copy=True, order='F', dtype=float)
    N = len(d)
    
    # Some helpers
    matrix_layout = 102
    double_p = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
    double_pp = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2)
    int_p = np.ctypeslib.ndpointer(dtype=np.int, ndim=1)
    
    # Define the function
    lapack = ctypes.CDLL(LAPACK_LIB)
    foo = lapack.LAPACKE_dstegr64_

    # int matrix_layout, char jobz,     char range,     lapack_int n, 
    # double *d,         double *e,     double vl,      double vu, 
    # lapack_int il,     lapack_int iu, double abstol,  lapack_int *m, 
    # double *w,         double *z,     lapack_int ldz, lapack_int *isuppz

    foo.restype = c_int
    foo.argtypes = [c_int,     c_char,    c_char,   c_int,
                    double_p,  double_p,  c_double, c_double,
                    c_int,     c_int,     c_double, int_p,
                    double_p,  double_p, c_int,    int_p]
    
    # matrix_layout, jobz, range, N, D, E, vl, vu, il, iu, abstol, m
    # 
    jobz = 'V'
    range = 'A'
    vl, vu = 0., 0.
    il, iu = 0, 0
    abstol = 0.
    m = np.zeros(N, dtype=np.int)
    w = np.zeros(N, dtype=np.float64)
    z = np.zeros(N, dtype=np.float64)
    ldz = N
    isuppz = np.zeros(2*N, dtype=np.int)

    info = foo(matrix_layout, jobz, range, N,
               D,             E,    vl,    vu,
               il,            iu,   abstol, m,
               w,             z,    ldz,    isuppz)
    
    # assert info == 0, "Got non-zero return value %d from LAPACK" % info
    # return WR

# ----------------------------------------------------------------------------

d = np.random.rand(100)
e = np.random.rand(99)

s3d_eig(d, e)
