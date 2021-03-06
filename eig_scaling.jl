include("utils.jl")
using Utils

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

        elseif problem == :lumped
            tic()
            Minv = Utils.lumped(M, -1.)
            A = Minv*A
            @assert typeof(A) == Tridiagonal{eltype(A)}
            A = full(A)
            eigw, eigv = eig(A)
            dt = toq()

        ###############################################################################
        # The one that matters: GEVP is transformed to EVP with SymTridiagonal martix #
        ###############################################################################
        elseif problem == :hermitian_lumped
            tic()
            Minv = Utils.lumped(M, -0.5)
            A = ⋆(A, Minv)
            @assert typeof(A) == SymTridiagonal{eltype(A)}
            eigw, eigv = eig(A)
            dt = toq()
        ###############################################################################

        elseif problem == :gen_hermitian_lumped
            tic()
            M = Symmetric(full(Utils.lumped(M)))
            eigw, eigv = eig(A, M)
            dt = toq()

        # The exact problem that we need to solve
        elseif problem == :gen_hermitian
            tic()
            eigw, eigv = eig(A, M)
            dt = toq()
        
        # Solve the genarized eigenvalue problem taking at least partially into
        # account SymTridiagonal properies of the matrix
        # NOTE It has worse convergence properties then eig(A, M)
        elseif problem == :gen_hermitian_stegr
            tic()
            eigw, eigv = eig(A, M)
            dt = toq()
        else
            @assert false
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
    cpu = first(Sys.cpu_info()).model
    mem = round(Int(Sys.total_memory())/10^9, 2)       # At GB
    f = "./data/jl_$(problem)_$(representation)_$(cpu)_$(mem).txt"
    save &&  writedlm(open(f, "w"), data)
end

# ----------------------------------------------------------------------------

if length(ARGS) in (1, 2)
    
    save = true
    problem = parse(first(ARGS))
    imax = length(ARGS) == 1 ? 14 : parse(last(ARGS))

    if problem == 0
        # LAPACK::GeneralMatrices::Eigenvalue dgeev
        println("eig(full(A))")
        scaling_eig(imax, :full, :hermitian, save)

    elseif problem == 1
        # LAPACK::SymmetricMatrices::Eigenvalue dsyevd -> same as python
        println("eig(Symmetric(A))")
        scaling_eig(imax, :Symmetric, :hermitian, save)

    elseif problem == 2
        # LAPACK routine for symmetric tridiagonal systems dstegt
        println("eig(SymTridiagonal(A))")
        scaling_eig(imax, :SymTridiagonal, :hermitian, save)

    elseif problem == 3
        # LAPACK::GeneralMatrices::Eigenvalue dggev
        println("eig(full(A), full(M))")
        scaling_eig(imax, :full, :gen_hermitian, save)

    elseif problem == 4
        # LAPACK::SymmetricMatrices::Eigenvalue dsygvd -> same as python
        println("eig(Symmetric(A), Symmetric(M))")
        scaling_eig(imax, :Symmetric, :gen_hermitian, save)

    elseif problem == 5
        # LAPACK::SymmetricMatrices::Eigenvalue dsygvd -> same as python
        println("eig(Symmetric(A), Symmetric(lumped(M)))")
        scaling_eig(imax, :Symmetric, :gen_hermitian_lumped, save)
    
    elseif problem == 6
        # LAPACK::GeneralMatrices::Eigenvalue dgeev
        println("eig(inv(lumped(M))*A)")
        scaling_eig(imax, :SymTridiagonal, :lumped, save)

    elseif problem == 7
        # LAPACK::SymmetricMatrices::Eigenvalue dsyevd -> same as python
        println("eig(A symmult lumped(M, -0.5))")
        scaling_eig(imax, :SymTridiagonal, :hermitian_lumped, save)

    # Uses partially the symtridiagonal properties
    elseif problem == 8
        # LAPACK::SymmetricMatrices::Eigenvalue dsyevd -> same as python
        println("eig(A::SymTridiagonal, M:SymTridiagonal)")
        scaling_eig(imax, :SymTridiagonal, :gen_hermitian_stegr, save)

    end
end
