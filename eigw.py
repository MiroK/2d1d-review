# Scaling of eigvalsh
from scipy.sparse import diags
from scipy.linalg import toeplitz, eigh, eig
import numpy as np
import subprocess
import psutil
import time


def system0(n):
    import dolfin as df
    mesh = df.UnitIntervalMesh(n)
    V = df.FunctionSpace(mesh, 'CG', 1)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    bc = df.DirichletBC(V, df.Constant(0), 'on_boundary')

    a = df.inner(df.grad(u), df.grad(v))*df.dx
    m = df.inner(u, v)*df.dx
    L = df.inner(df.Constant(0), v)*df.dx

    A, _ = df.assemble_system(a, L, bc)
    M, _ = df.ssemble_system(m, L, bc)

    return A, M


def system(n):
    h = 1./n

    dA = np.r_[1., 2./h*np.ones(n-1), 1.]
    uA = np.r_[0, -1./h*np.ones(n-2), 0.]
    A = diags([uA, dA, uA], [-1, 0, 1])

    dM = np.r_[1., 4*h/6*np.ones(n-1), 1.]
    uM = np.r_[0, h/6.*np.ones(n-2), 0.]
    M = diags([uM, dM, uM], [-1, 0, 1])

    return A, M


def test_system():
    for n in [10, 100, 1000, 10000]:
        x = np.random.rand(n+1)

        A0, M0 = system0(n)
        A, M = system(n)

        assert np.linalg.norm(A.dot(x)-A0.array().dot(x))/(n+1) < 1E-10
        assert np.linalg.norm(M.dot(x)-M0.array().dot(x))/(n+1) < 1E-10


def lump(mat, power=1.):
    d = np.sum(mat, 1)
    d = d**power
    return np.diag(d)


def cpu_type():
    all_info = subprocess.check_output('cat /proc/cpuinfo', shell=True).strip()
    cpus = [line.split(':')[-1].strip()
            for line in all_info.split("\n") if 'model name' in line]
    return cpus[0]


def mem_total():
    return psutil.virtual_memory().total/10.**9


def scaling_eigvalsh(problem='hermitian', imax=15):
    dt0, rate = -1, np.nan
    data = []
    for i in range(1, imax):
        n = 2**i
        A, M = system(n)
        A, M = A.toarray(), M.toarray()
        
        if problem == 'hermitian':
            # Ends up calling LAPACK::ssyevd
            t0 = time.time()
            eigw, eigv = eigh(A)
            dt = time.time() - t0
        
        elif problem == 'lumped':
            # Lump the mass matrix and take its inverse to the other side
            # inv(lumped(M))*A is a tridiagonal matrix
            t0 = time.time()
            Minv = lump(M, -1.)
            A = Minv.dot(A)
            eigw, eigv = eig(A)
            dt = time.time() - t0

        elif problem == 'hermitian_lumped':
            # lumped(M)^{-1/2}*A*lumped(M)^{-1/2} is a symmetric tridiagonal
            # matrix. We use specialized routine to compute its eigen
            # factorization
            from lapack_stegr import s3d_eig

            # We know this can be done efficiently as in julia and so we don't
            # measure it
            M = lump(M, -0.5)                            #
            A = M.dot(A.dot(M))                          #
            d, u = np.diagonal(A, 0), np.diagonal(A, 1)  # 

            t0 = time.time()
            eigw, eigv = s3d_eig(d, u)
            dt = time.time() - t0

        elif problem == 'gen_hermitian':
            # Ends up calling LAPACK::ssygvd
            t0 = time.time()
            eigw, eigv = eigh(A, M)
            dt = time.time() - t0

        elif problem == 'gen_hermitian_lumped':
            # Lump the mass matrix and keep it on the right hand size - solve a
            # generalized eigenvalue problem
            t0 = time.time()
            M = lump(M)
            eigw, eigv = eigh(A, M)
            dt = time.time() - t0

        else:
            raise ValueError

        lmin, lmax = np.min(eigw), np.max(eigw)

        if dt0 > 0:
            rate = np.log(dt/dt0)/np.log(2)
            fmt = 'size %d took %g s rate = %.2f, lmin = %g, lmax = %g'
            print fmt % (A.shape[0], dt, rate, lmin, lmax)

        dt0 = dt
        data.append([A.shape[0], dt, rate, lmin, lmax])
    np.savetxt('./data/py_%s_%s_%.2f.txt' % (problem, cpu_type(), mem_total()),
               np.array(data),
               header='Ashape, CPU time, rate, lmin, lmax')

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    assert len(sys.argv) in (2, 3)
    i = int(sys.argv[1])

    if i == -1:
        print 'Testing'
        test_system()
        A, M = system(1000)
        # print A.toarray()
        # print M.toarray()
        A, M = A.toarray(), M.toarray()
        eigw, eigv = eigh(A, M)
        # for i in range(len(eigw)):
        #     w = eigw[i]
        #     v = eigv[:, i]
        #     error = np.linalg.norm(A.dot(v) - w*M.dot(v))
        #     assert all(abs(np.sum(v*(M.dot(eigv[:, j])))) < 1E-13 
        #                for j in range(i))
        #     print w, error, np.sum(v*(M.dot(v)))
        #     # println(error)
        print min(eigw), max(eigw)

    elif i == 0:
        print 'Scaling of eigh(A)'
        imax = 15 if len(sys.argv) == 2 else int(sys.argv[2])
        scaling_eigvalsh(problem='hermitian', imax=imax)

    elif i == 1:
        print 'Scaling of eigh(A, M)'
        imax = 15 if len(sys.argv) == 2 else int(sys.argv[2])
        scaling_eigvalsh(problem='gen_hermitian', imax=imax)

    elif i == 2:
        print 'Scaling of eigh(A, lumped(M))'
        imax = 15 if len(sys.argv) == 2 else int(sys.argv[2])
        scaling_eigvalsh(problem='gen_hermitian_lumped', imax=imax)

    elif i == 3:
        print 'Scaling of eig(inv(lumped(M))*A)'
        imax = 15 if len(sys.argv) == 2 else int(sys.argv[2])
        scaling_eigvalsh(problem='lumped', imax=imax)
    ###############################################################################
    # The one that matters: GEVP is transformed to EVP with SymTridiagonal martix #
    ###############################################################################
    elif i == 4:
        print 'Scaling of eigh(Mi*A*Mi)'
        imax = 15 if len(sys.argv) == 2 else int(sys.argv[2])
        scaling_eigvalsh(problem='hermitian_lumped', imax=imax)
    ###############################################################################
