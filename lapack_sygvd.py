import numpy as np
import ctypes
from ctypes import c_int, c_char, c_double

# Select which blas to use. Note that the LIB path is hardcoded
    impl = 'mkl'

# Julia's openblas
if impl == 'jl':
    LAPACK_LIB = '/mn/anatu/ansatte-u8/mirok/Documents/Programming/julia/usr/lib/libopenblas64_.so' 
# Open blas from hashdist
elif impl == 'py':
    LAPACK_LIB = '/mn/anatu/ansatte-u8/mirok/.hashdist/bld/openblas/c7qemuijpr3x/lib/libopenblas.so'
# Intel
else:
    LAPACK_LIB ='/opt/uio/modules/packages/intel/14.0.2.144/mkl/lib/intel64/libmkl_rt.so'

# Define the function
lapack = ctypes.CDLL(LAPACK_LIB)

matrix_layout = 101
double_p = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
double_pp = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2)

# Julia has special names
foo = lapack.LAPACKE_dsygvd64_ if impl == 'jl' else lapack.LAPACKE_dsygvd
# int matrix_layout, lapack_int itype, char jobz, char uplo, 
# lapack_int n, 
# double *a, lapack_int lda,
# double *b, lapack_int ldb,
# double *w
foo.restype = c_int
foo.argtypes = [c_int,     c_int,   c_char,    c_char,
                c_int,
                double_pp, c_int,
                double_pp, c_int,
                double_p]


def dsygvd(jobz, A, B):
    '''
    Eigen factorization of Ax = lambda Bx with A symmetric, B symmetric-definite. 
    Computed by LAPACKE.dsygvd.
    '''
    assert issquare(A) and issquare(B)
    assert jobz in ('V', 'N')

    itype = 1
    # Copy so that d, u are not changed
    A = np.array(A, copy=True, order='C', dtype=float)
    B = np.array(B, copy=True, order='C', dtype=float)
    N = len(A)
    w = np.zeros(N, dtype=np.float64)
    
    uplo = 'U'
    lda = N
    ldb = N

    info = foo(matrix_layout, itype, jobz, uplo,
               N,
               A, lda,
               B, ldb, 
               w)

    assert info == 0, 'LAPACK info %d' % info
    return (w, A) if jobz == 'V' else w

# --------------------


def issquare(A): 
    '''Check if A is a square matrix.'''
    return A.ndim == 2 and len(set(A.shape)) == 1


def issym(A, tol):
    '''See if matrix is symmetric within tol.'''
    nrows = A.shape[0]
    assert issquare(A)
    x = np.random.rand(nrows)
    return np.linalg.norm(A.dot(x) - A.T.dot(x)) < tol


def sym_geig(A, B, check_sym=-1):
    '''Eigen factorization of Ax = lambda Bx with A symmetric, B spd.'''
    if check_sym > 0: assert issym(A, check_sym) and issym(B, check_sym)
    return dsygvd('V', A, B)


def sym_geigvals(A, B, check_sym=-1):
    '''Eigenvalues of Ax = lambda Bx with A symmetric, B spd.'''
    if check_sym > 0: assert issym(A, check_sym) and issym(B, check_sym)
    return dsygvd('N', A, B)


def test():
    from scipy.sparse import diags
    from scipy.linalg import eigh
    from eigw import system
    import time

    nrepeats = 1
    ns = [2**i for i in [10, 11, 12]]
    times = np.zeros((len(ns), 1+3+3))
    times[:, 0] = ns
    for row, n in enumerate(ns):
        ntimes, ntimes0 = [], []
        A, B = system(n)
        A, B = A.toarray(), B.toarray()
        
        for i in range(nrepeats):
            # A will be symmetric, B symmertic positive definite

            # Call lapack on symmetric generalized eigenvalue problem
            t = time.time()
            w, v = sym_geig(A, B)
            t = time.time()-t
            ntimes.append(t)

            # Expensive but important metrics - works 
            # werror = np.max([np.linalg.norm(A.dot(v[:, col])-w[col]*B.dot(v[:, col]))
            #                  for col in range(len(w))])
            # onerror = np.array([np.sum(x*B.dot(x)) for x in v.T])
            # onerror = np.linalg.norm(onerror - np.ones_like(onerror))

            # ogerror = np.array([np.sum(v[:, ii]*B.dot(v[:, jj]))
            #                     for ii in range(len(w)) for jj in range(ii+1, len(w))])
            # ogerror = np.linalg.norm(ogerror)

            # print werror, onerror, ogerror
            
            # Call scipy on ...
            t0 = time.time()
            w0, v0 = eigh(A, B)
            t0 = time.time()-t0
            ntimes0.append(t0)

            # print np.linalg.norm(w-w0)
            # print np.linalg.norm(v-v0)

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

    test()

    # NOTE python/anaconda-haak*      is as fast as calling lapack like above
    #      hashdist fenics.           same holds
    # This is for 4096 2x slower than julia!
