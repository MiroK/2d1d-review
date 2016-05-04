from dolfin import *

def extend(n_cells, k):
    '''
    Harmonic extension of f=sin(k*pi*x) by

    ------------------
    |                |
    |                |
    |    \Gamma      |
    |----------------|
    |                |
    |                | \partial\Omega
    |                |
    |----------------|

    -Delta u = u
           u = 0 on \partial\Omega\setminus \Gamma
           u = f on \Gamma
    '''
    assert n_cells % 2 == 0
    f = Expression('sin(k*pi*x[0])', k=k)

    mesh = UnitSquareMesh(n_cells, n_cells)
    mesh = RectangleMesh(Point(0, 0), Point(1, 0.5), n_cells, n_cells/2)

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    gamma = FacetFunction('size_t', mesh, 0)
    CompiledSubDomain('near(x[1], 0.5)').mark(gamma, 1)
    
    bc_out = DirichletBC(V, Constant(0), 'on_boundary')
    bc_in = DirichletBC(V, f, gamma, 1)
    bcs = [bc_out, bc_in]

    a = inner(grad(u), grad(v))*dx
    L = inner(Constant(0), v)*dx
    
    uh = Function(V)
    solve(a == L, uh, bcs)

    return uh

# ----------------------------------------------------------------------------

if __name__ == '__main__':

    k = 2
    plot(extend(n_cells=16, k=k))
    plot(extend(n_cells=32, k=k))
    plot(extend(n_cells=64, k=k))
    interactive()
