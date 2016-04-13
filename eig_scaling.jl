include("utils.jl")
import Utils

"""
Scaling of algorithms for (generalized)hermitian eiganvalue problems. Julia
selects the algorithm based on typeof matrix.
"""
function scaling_eig(imax, representation, problem=:hermitian, save=false)
    data = zeros(imax, 5)
    dt0, rate = -1., -1.
    for i in 1:imax
        n = 2^i
        A, M = Utils.matrices(n, representation)

        if problem == :hermitian
            tic()
            eigw, eigv = eig(A)
            dt = toq()
        else
            tic()
            eigw, eigv = eig(A, M)

            dt = toq()
        end

        lmin, lmax = minimum(eigw), maximum(eigw)
        @assert lmin > 0

        if dt0 > 0
            rate = log(dt/dt0)/log(2)
            @printf("size %d took %g s rate %.2f, lmin = %g, lmax = %g\n", size(A, 1), dt, rate, lmin, lmax)
        end

        dt0 = dt
        data[i, :] = [size(A, 1), dt, rate, lmin, lmax]
    end
    save && writedlm(open("./data/jl_$(problem)_$(representation).txt", "w"), data)
end

# ----------------------------------------------------------------------------

if length(ARGS) in (1, 2)
    
    save = false
    problem = parse(first(ARGS))
    imax = length(ARGS) == 1 ? 14 : parse(last(ARGS))

    if problem == 0
        # LAPACK::GeneralMatrices::Eigenvalue dgeev
        println("eig(full(A))")
        scaling_eig(imax, :full, save)

    elseif problem == 1
        # LAPACK::SymmetricMatrices::Eigenvalue dsyevd -> same as python
        println("eig(Symmetric(A))")
        scaling_eig(imax, :Symmetric, save)

    elseif problem == 2
        # LAPACK routine for symmetric tridiagonal systems dstegt
        println("eig(SymTridiagonal(A))")
        scaling_eig(imax, :SymTridiagonal, save)

    elseif problem == 3
        # LAPACK::GeneralMatrices::Eigenvalue dggev
        println("eig(full(A), full(M))")
        scaling_eig(imax, :full, :gen_hermitian, save)

    elseif problem == 4
        # LAPACK::SymmetricMatrices::Eigenvalue dsygvd -> same as python
        println("eig(Symmetric(A), Symmetric(M))")
        scaling_eig(imax, :Symmetric, :gen_hermitian, save)
    end
end
