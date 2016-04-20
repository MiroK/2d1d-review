import numpy as np

# Select which blas to use. Note that the LIB path is hardcoded
impl = 'jl'

# Julia's openblas
if impl == 'jl':
    LAPACK_LIB = '/mn/anatu/ansatte-u8/mirok/Documents/Programming/julia/usr/lib/libopenblas64_.so' 
# Open blas from hashdist
elif impl == 'py':
    LAPACK_LIB = '/mn/anatu/ansatte-u8/mirok/.hashdist/bld/openblas/c7qemuijpr3x/lib/libopenblas.so'
# Intel
else:
    LAPACK_LIB ='/opt/uio/modules/packages/intel/14.0.2.144/mkl/lib/intel64/libmkl_rt.so'


def dstegr(jobz, d, u):
    '''Eigen factorization of symmetric tridiagonal matrix with LAPACK.DSTEGR.'''
    import ctypes
    from ctypes import c_int, c_char, c_double
    assert jobz in ('V', 'N')
    assert len(d) == len(u)+1

    # Copy so that d, u are not changed
    D = np.array(d, copy=True, order='C', dtype=float)
    # Do like julia and prepend zero
    E = np.r_[np.array(u, copy=True, order='C', dtype=float), 0]
    N = len(d)

    # Some helpers
    matrix_layout = 101
    double_p = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
    double_pp = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2)
    int_p = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1)
    
    # Define the function
    lapack = ctypes.CDLL(LAPACK_LIB)
    # Julia has spacial names
    foo = lapack.LAPACKE_dstegr64_ if impl == 'jl' else lapack.LAPACKE_dstegr

    # int matrix_layout, char jobz,     char range,     lapack_int n, 
    # double *d,         double *e,     double vl,      double vu, 
    # lapack_int il,     lapack_int iu, double abstol,  lapack_int *m, 
    # double *w,         double *z,     lapack_int ldz, lapack_int *isuppz

    foo.restype = c_int
    foo.argtypes = [c_int,     c_char,    c_char,   c_int,
                    double_p,  double_p,  c_double, c_double,
                    c_int,     c_int,     c_double, int_p,
                    double_p,  double_pp, c_int,    int_p]
    
    range = 'A'       # All eigenvalues
    vl, vu = 0., 0.
    il, iu = 0, 0
    abstol = 0.
    m = np.zeros(1, dtype=np.int64)
    w = np.zeros(N, dtype=np.float64)
    z = np.zeros((N, N), dtype=np.float64)
    ldz = N
    isuppz = np.zeros(2*N, dtype=np.int64)

    info = foo(matrix_layout, jobz, range, N,
               D,             E,    vl,    vu,
               il,            iu,   abstol, m,
               w,             z,    ldz,    isuppz)

    assert info == 0, 'LAPACK info %d' % info
    return (w, z) if jobz == 'V' else w


def s3d_eig(d, u):
    '''Eigen factorization of symmetric tridiagonal matrix with LAPACK.DSTEGR.'''
    return dstegr('V', d, u)


def s3d_eigvals(d, u):
    '''Eigenvalues of symmetric tridiagonal matrix with LAPACK.DSTEGR.'''
    return dstegr('N', d, u)

def test():
    from scipy.sparse import diags
    from scipy.linalg import eigh
    import time

    ns = [2**i for i in [10, 11, 12, 13]]
    times = np.zeros((len(ns), 1+3+3))
    times[:, 0] = ns
    for row, n in enumerate(ns):
        ntimes, ntimes0 = [], []
        for i in [1, 2, 3]:
            d = np.random.rand(n)+1
            e = np.zeros(n-1)

            t = time.time()
            w, v = s3d_eig(d, e)
            t = time.time()-t
            ntimes.append(t)

            A = diags([e, d, e], [-1, 0, 1]).toarray()
            t0 = time.time()
            w0, v0 = eigh(A)
            t0 = time.time()-t0
            ntimes0.append(t0)
            
            assert np.linalg.norm(w-w0) < 1E-13
            assert np.linalg.norm(v-v0) < 1E-13

        times[row, 1:4] = [np.min(ntimes), np.average(ntimes), np.max(ntimes)]
        times[row, 4:] = [np.min(ntimes0), np.average(ntimes0), np.max(ntimes0)]

    for row in times:
        row_ = [row[0]] + sum(([row[i], row[i+3]/row[i]] for i in (1, 2, 3)), [])
        print '%d min=%.3f(%.2f) avg=%.3f(%.2f) max=%.3f(%.2f)' % tuple(row_)
    print

    np.set_printoptions(formatter={'all': lambda v: '%.2f' % v})
    print times

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from scipy.sparse import diags
    from scipy.linalg import eigh
    import time

    ns = [2**i for i in [10, 11, 12, 13]]
    times = np.zeros((len(ns), 1+3))
    times[:, 0] = ns
    for row, n in enumerate(ns):
        ntimes, ntimes0 = [], []
        for i in [1, 2, 3]:
            d = np.random.rand(n)+1
            e = np.random.rand(n-1)

            t = time.time()
            w, v = s3d_eig(d, e)
            t = time.time()-t
            ntimes.append(t)

        times[row, 1:] = [np.min(ntimes), np.average(ntimes), np.max(ntimes)]
    
    print impl
    for row in times:
        print '%d min=%.3f avg=%.3f max=%.3f' % tuple(row)
    print
