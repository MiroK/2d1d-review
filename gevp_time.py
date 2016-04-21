#
# We are interested in how much time GEVP solve takes on relevant 1d meshes
#
from dolfin import Mesh, MeshFunction, UnitSquareMesh, EdgeFunction, CompiledSubDomain
from dolfin import FunctionSpace, TrialFunction, TestFunction, DirichletBC
from dolfin import inner, grad, Constant, dx, assemble_system
from mesh_extraction import interval_mesh_from_edge_f
from scipy.linalg import eigh
from dolfin import Timer
import numpy as np

from lapack_stegr import s3d_eig
from eigw import lump, cpu_type, mem_total


def get_1d_matrices(mesh_, N, root=''):
    '''Given mesh construct 1d matrices for GEVP.'''
    if not isinstance(N, (int, float)):
        assert root
        return all([get_1d_matrices(mesh_, n, root) == 0 for n in N])

    # Zig zag mesh
    if mesh_ == 'nonuniform':
        mesh_dir = '../plate-beam/py/fem_new/meshes'
        mesh = 'Pb_zig_zag_bif'
        mesh2d = '%s/%s_%d.xml.gz' % (mesh_dir, mesh, N)
        mesh1d = '%s/%s_%d_facet_region.xml.gz' % (mesh_dir, mesh, N)
        mesh2d = Mesh(mesh2d)
        mesh1d = MeshFunction('size_t', mesh2d, mesh1d)
    # Structured meshes
    else:
        N = int(N)
        mesh2d = UnitSquareMesh(N, N)
        mesh1d = EdgeFunction('size_t', mesh2d, 0)
        # Beam at y = 0.5
        CompiledSubDomain('near(x[1], 0.5, 1E-10)').mark(mesh1d, 1)

    print FunctionSpace(mesh2d, 'CG', 1).dim()

    # Extract 1d
    import sys; sys.setrecursionlimit(20000);
    mesh = interval_mesh_from_edge_f(mesh2d, mesh1d, 1)[0].pop()

    # Assemble
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(0), 'on_boundary')

    a = inner(grad(u), grad(v))*dx
    m = inner(u, v)*dx
    L = inner(Constant(0), v)*dx

    A, _ = assemble_system(a, L, bc)
    M, _ = assemble_system(m, L, bc)

    A, M = A.array(), M.array()

    if root:
        dA = np.diagonal(A, 0)
        uA = np.r_[np.diagonal(A, 1), 0]
        A = np.c_[dA, uA]

        dM = np.diagonal(M, 0)
        uM = np.r_[np.diagonal(M, 1), 0]
        M = np.c_[dM, uM]

        header = 'main and upper diagonals of A, M'
        f = '_'.join([mesh_, str(N)])
        import os
        f = os.path.join(root, f)
        np.savetxt(f, np.c_[A, M], header=header)
        return 0
    else:
        return A, M


def read_matrices(mesh, root):
    import os
    from scipy.sparse import diags

    files = os.listdir(root)
    files = filter(lambda f: f.startswith(mesh), files)
    files = sorted(files, key=lambda v: int(v.split('_')[-1]))

    for f in files:
        data = np.loadtxt(os.path.join(root, f))
        d, u = data[:, 0], data[:-1, 1]
        A = diags([u, d, u], [-1, 0, 1]).toarray()
        d, u = data[:, 2], data[:-1, 3]
        M = diags([u, d, u], [-1, 0, 1]).toarray()
        yield A, M


def python_timings(mesh, Nrange, read_matrix=False):
    '''Run across meshes solving lumped EVP and recording their sizes and exec time.'''
    # We know from julia that this idea(lumping) works. 
    # So we are only after timing
    data = []

    if read_matrix:
        matrices = read_matrices(mesh, root='./jl_matrices')
    else:
        matrices = (get_1d_matrices(mesh, N) for N in Nrange)

    for A, M in matrices:
        row = [A.shape[0]]

        M0 = lump(M, -0.5)                             #
        A0 = M0.dot(A.dot(M0))                           #
        d, u = np.diagonal(A0, 0), np.diagonal(A0, 1)   # 

        t = Timer('EVP')
        eigw, eigv = s3d_eig(d, u)
        row.append(t.stop())

        # Assembling the preconditioner
        t = Timer('ASSEMBLE')
        H = eigv.dot(np.diag(eigw**-0.5).dot(eigv.T))
        row.append(t.stop())

        # Action of preconditioner(matrix)
        x = np.random.rand(H.shape[1])
        t = Timer('ACTION')
        y = H*x
        row.append(t.stop())

        # The original GEVP
        t = Timer('GEVP')
        eigw, eigv = eigh(A, M)
        row.append(t.stop())

        print row
        data.append(row)

    return data

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    # Generate uniform matrices for julia
    # mesh, Ns = 'uniform', [2**i for i in range(2, 12)] + [2**i for i in (11.5, 11.7)]
    # print get_1d_matrices(mesh, Ns, root='./jl_matrices')

    # Generate nonuniform matrices for julia
    # mesh, Ns = 'nonuniform', range(8)
    # print get_1d_matrices(mesh, Ns, root='./jl_matrices')
    read_matrix = False

    if False:
        data = python_timings('uniform',
                              [2**i for i in range(2, 12)] + [2**i for i in (11.5, 11.7)],
                              read_matrix)

        np.savetxt('./data/py_uniform_%d_@%s_%.2f_evp' % (read_matrix, cpu_type(), mem_total()),
                   data,
                   header='size, EVP, ASSEMBLE, ACTION, GEVP. Julia has c, C')

    if True:
        data = python_timings('nonuniform', range(8), read_matrix)
        np.savetxt('./data/py_nonuniform_%d_@%s_%.2f_evp' % (read_matrix, cpu_type(), mem_total()),
                   data,
                   header='size, EVP, ASSEMBLE, ACTION, GEVP. Julia has c, C')

