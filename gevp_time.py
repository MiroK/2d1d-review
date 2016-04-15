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


def get_1d_matrices(mesh, N):
    '''Given mesh construct 1d matrices for GEVP.'''
    # Zig zag mesh
    if mesh == 'nonuniform':
        mesh = 'Pb_zig_zag_bif'
        mesh2d = './meshes/%s_%d.xml' % (mesh, N)
        mesh1d = './meshes/%s_%d_facet_region.xml' % (mesh, N)
        mesh2d = Mesh(mesh2d)
        mesh1d = MeshFunction('size_t', mesh2d, mesh1d)
    # Structured meshes
    else:
        mesh2d = UnitSquareMesh(2**N, 2**N)
        mesh1d = EdgeFunction('size_t', mesh2d, 0)
        # Beam at y = 0.5
        CompiledSubDomain('near(x[1], 0.5, 1E-10)').mark(mesh1d, 1)

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

    return A.array(), M.array()


def python_timings(mesh, Nrange):
    '''Run acorss meshes solving GEVP and recording their sizes and exec time.'''
    times = []
    sizes = []

    for A, M in (get_1d_matrices(mesh, N) for N in Nrange):
        sizes.append(A.shape[0])

        ts = []
        # Solving the eigenvalue problem
        t = Timer('GEVP')
        eigw, eigv = eigh(A, M)
        ts.append(t.stop())

        # Assembling the preconditioner
        t = Timer('ASSEMBLE')
        H = eigv.dot(np.diag(eigw**-0.5).dot(eigv.T))
        ts.append(t.stop())

        # Action of preconditioner(matrix)
        x = np.random.rand(H.shape[1])
        t = Timer('ACTION')
        y = H*x
        ts.append(t.stop())

        times.append(ts)

        print sizes[-1], times[-1]

    record = np.c_[sizes, times]
    return record

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    print python_timings('uniform', range(2, 11))

    # Add lumping how
    # Add saving





