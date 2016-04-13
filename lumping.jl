include("utils.jl")
using Utils

"""Ax = lambda*B*x vs Ax = lambda*diag(B)*x"""
function lumping_effect(imax)
    dt0, rate = -1., -1.
    for i in 1:imax
        n = 2^i
        A, M = Utils.matrices(n, :SymTridiagonal)
        iMl = Diagonal(1./Utils.lumped(M).diag)
        iMlA = iMl*A
        # FIXME: this now correctly return Tridigonal matrix but there is no
        # specialized factorization for that so full
        eigw, _ = eig(full(iMlA))

        A, M = Utils.matrices(n, :Symmetric)
        eigw0, _ = eig(A, M)

        lmin, lmax = minimum(eigw), maximum(eigw)
        lmin0, lmax0 = minimum(eigw0), maximum(eigw0)

        println("$(n) $(lmin0/lmin) $(lmax0/lmax)")

    end
end

# ----------------------------------------------------------------------------

if length(ARGS) == 1
    imax = parse(first(ARGS))
    lumping_effect(imax)
end
