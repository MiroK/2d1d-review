from dolfin import *
from block.algebraic.petsc import AMG
import numpy as np

# How much time AMG takes on the meshes where we do gevp timing

mesh_, Ns = 'uniform', [2**i for i in range(2, 12)] + [2**i for i in (11.5, 11.7)] 
#mesh_, Ns = 'nonuniform', range(8)


data = []
for N in Ns:
    if mesh_ == 'nonuniform':
        mesh_dir = '../plate-beam/py/fem_new/meshes'
        mesh = 'Pb_zig_zag_bif'
        mesh = '%s/%s_%d.xml.gz' % (mesh_dir, mesh, N)
        mesh = Mesh(mesh)
    else:
        N = int(N)
        mesh = UnitSquareMesh(N, N)

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(0), 'on_boundary')

    a = inner(grad(u), grad(v))*dx
    m = inner(u, v)*dx
    L = inner(Constant(1), v)*dx

    A, b = assemble_system(a, L, bc)

    timer = Timer('AMG')
    P = AMG(A)
    amg_time = timer.stop()

    timer = Timer('AMG_action')
    x = P*b
    amg_action = timer.stop()

    data.append([V.dim(), amg_time, amg_action])
    print data[-1]

data = np.array(data)
print data


np.savetxt('./data/py_%s_amg' % mesh_, data,
           header='size(A) time to construct P=AMG(A) and its action')
