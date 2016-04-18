module Utils

export ⋆

using Base.Test

"""
Stiffness and mass matrices of UnitIntervalMesh with CG1 and Dirichlet bcs
on both ends.
"""
function matrices(n, representation)
    h = 1./n

    dA = Vector{Float64}([1.; fill(2./h, n-1); 1.])
    uA = Vector{Float64}([0; fill(-1./h, n-2); 0.])
    A = SymTridiagonal(dA, uA)

    dM = Vector{Float64}([1.; fill(4*h/6, n-1); 1.])
    uM = Vector{Float64}([0; fill(h/6., n-2); 0.])
    M = SymTridiagonal(dM, uM)

    if representation != :SymTridiagonal
        (A, M) = map(full, (A, M))
        if representation == :Symmetric
            (A, M) = map(Symmetric, (A, M))
        end
    end

    (A, M)
end

"""Lumping for symmetric tridiagonal matrix."""
function lumped{T<:Number}(M::SymTridiagonal{T}, power::Real=1.)
    dv, ev = M.dv, M.ev
    d = Vector{T}([first(dv)+first(ev); ev[1:end-1]+dv[2:end-1]+ev[2:end]; last(dv)+last(ev)])
    d = d.^power
    Diagonal(d)
end

"""Lumping for general matrix."""
function lumped{T<:Number}(M::Matrix{T}, power::Real=1.)
    d = sum(M, 2)
    d = reshape(d, length(d))
    d = d.^power
    Diagonal(d)
end

"""Lumping for symmetric matrix."""
function lumped{T<:Number}(M::Symmetric{T}, power::Real=1.)
    lumped(full(M), power)
end

# Convenience inner product
inner{T<:Number, S<:Number}(u::Vector{T}, v::Vector{S}) = sum(u.*v)

import Base: start, next, done, length
# Iterator over colums of a matrix
immutable cols{T<:Real}
    A::Matrix{T}
    indices::OrdinalRange{Int, Int} 
end

# All columns
function cols{T}(A::Matrix{T})
    indices = 1:size(A, 2)
    cols(A, indices)
end

# Iterator protocol
start(it::cols) = start(it.indices)

function next(it::cols, state)
    col, state = next(it.indices, state)
    (it.A[:, col], state)
end

done(it::cols, state) = done(it.indices, state) 

length(it::cols) = length(it.indices)

# Iterator over rows of a matrix. FIXME: rewrite as such that typealias rows{T}
# is axis_iterator{1}
immutable rows{T<:Real}
    A::Matrix{T}
    indices::OrdinalRange{Int, Int} 
end

# All columns
function rows{T}(A::Matrix{T})
    indices = 1:size(A, 1)
    rows(A, indices)
end

# Iterator protocol
start(it::rows) = start(it.indices)

function next(it::rows, state)
    row, state = next(it.indices, state)
    (it.A[row, :], state)
end

done(it::rows, state) = done(it.indices, state) 

length(it::rows) = length(it.indices)

# This is motivation for the defs below
# FIXME is there a a clever(fast) way of computing eigs here? Note that this
# comes from Ax = lambda diag(B)*x (1) where A, B are Sym3 so equally good
# answer for us is to solve (1) w/out ruining the symmetry. Finally the best
# answer is to solve (1) with full B. 

import Base.*


# Specialize Tridiagonal*Diagonal -> Tridiagonal
function *{T<:Number, S<:Number}(A::Tridiagonal{T}, B::Diagonal{S})
    dl, d, du = A.dl, A.d, A.du
    dB = B.diag

    upper = du.*dB[2:end]
    diag = d.*dB
    lower = dl.*dB[1:end-1]

    Tridiagonal(lower, diag, upper)
end

# Specialize Diagonal*Tridiagonal -> Tridiagonal
function *{T<:Number, S<:Number}(A::Diagonal{T}, B::Tridiagonal{S})
    dA = A.diag
    dl, d, du = B.dl, B.d, B.du

    upper = dA[1:end-1].*du
    diag = dA.*d
    lower = dA[2:end].*dl

    Tridiagonal(lower, diag, upper)
end

# Specialize product for Diag*SymTridiagonal -> Tridiagonal
function *{T<:Number, S<:Number}(A::Diagonal{T}, B::SymTridiagonal{S})
    d = A.diag
    dv, ev = B.dv, B.ev

    upper = d[1:end-1].*ev
    diag = d.*dv
    lower = d[2:end].*ev
    Tridiagonal(lower, diag, upper)
end

# Specialize product for SymTridiagonal*Diag -> Tridiagonal
function *{T<:Number, S<:Number}(A::SymTridiagonal{T}, B::Diagonal{S})
    d = B.diag
    dv, ev = A.dv, A.ev

    upper = d[2:end].*ev
    diag = d.*dv
    lower = d[1:end-1].*ev
    Tridiagonal(lower, diag, upper)
end

# Let's make special inner product for D*A*D.T with diagonal D and symmetric
# tridiagonal A
function ⋆{T<:Number, S<:Number}(A::SymTridiagonal{T}, B::Diagonal{S})
    dv, ev = A.dv, A.ev
    d = B.diag

    upper = ev.*d[2:end].*d[1:end-1]
    diag = dv.*d.*d
    SymTridiagonal(diag, upper)
end

# Mat*SymTridiagonal*Mat.T is SymTridiagonal
# TODO + test

# Probability norm. Generate a bunch of random vectors and see if about the l^2
# norm of mat*vec
function p_norm(mat::Matrix, nvecs=0)
    nvecs = (nvecs == 0) ? round(Int, sqrt(size(mat, 1))) : nvecs

    errors = map(1:nvecs) do i
        begin
            r = rand(size(mat, 2))
            norm(mat*r)/nvecs
        end
    end

    avg = sum(errors)/nvecs
    max = maximum(errors)
    min = minimum(errors)

    return (min, avg, max, length(errors))


    errors = zeros(Float64, nvecs)
    for i in 1:nvecs
        errors[i] 
    end
end


# Operator norm |A| = sup (x*A*x)/(x*M*x)
op_norm(mat, M) = maximum(abs(first(eigs(mat, M, which=:LM, ritzvec=false))))
# op_norm(mat, M) = maximum(abs(first(eig(mat, M))))

# With idenity
op_norm(mat) = eigmax(full(mat))

# Compare with identity matrix
iseye(mat, tol=1E-10) = size(mat, 1) == size(mat, 2) && norm(mat-eye(size(mat, 1))) < tol

# Compare with zeros
iszeros(mat, tol=1E-10) = norm(mat) < tol

########
# TESTS
########
function test()
    # Lumping
    dv = rand(20)
    ev = rand(19)
    M = SymTridiagonal(dv, ev)
    d = lumped(M).diag
    M = full(M)
    d0 = [sum(M[i, :]) for i in 1:size(M, 1)]
    @test_approx_eq_eps norm(d-d0) 0. 1E-13

    # Multiplication
    A = Diagonal(rand(10))
    B = SymTridiagonal(rand(10), rand(9))
    C = A*B
    @test typeof(C) == Tridiagonal{eltype(A)}
    C0 = full(A)*full(B)
    @test_approx_eq_eps norm(C-C0) 0. 1E-13

    C = B*A
    @test typeof(C) == Tridiagonal{eltype(A)}
    C0 = full(B)*full(A)
    @test_approx_eq_eps norm(C-C0) 0. 1E-13

    C = B⋆A
    @test typeof(C) == SymTridiagonal{eltype(A)}
    C0 = A*(B*A)
    @test_approx_eq_eps norm(full(C-C0)) 0. 1E-13    
    # NOTE C-C0 is tridiag and svdvals! fails hence conversion to full is needed

    B = Tridiagonal(rand(9), rand(10), rand(9))
    C = A*B
    @test typeof(C) == Tridiagonal{eltype(A)}
    C0 = full(A)*full(B)
    @test_approx_eq_eps norm(C-C0) 0. 1E-13

    C = B*A
    @test typeof(C) == Tridiagonal{eltype(A)}
    C0 = full(B)*full(A)
    @test_approx_eq_eps norm(C-C0) 0. 1E-13

    # is*
    @test iseye(eye(10)) == true
    @test iszeros(zeros(10, 10)) == true
    @test iszeros(zeros(10)) == true

    # norms
    @test_approx_eq_eps op_norm(A) maximum(A.diag) 1E-13
    @test_approx_eq_eps op_norm(A, A) 1. 1E-13
    @test p_norm(zeros(9, 9)) == (0., 0., 0., 3)

    # iterators
    A = rand(10, 10)
    # cols all
    for (i, c) in enumerate(cols(A))
        @test_approx_eq norm(A[:, i]-c) 0.
        @test typeof(c) == Vector{Float64}
    end
    # cal subset
    count = 0
    for (i, c) in zip(1:5:2, cols(A, 1:5:2))
        @test_approx_eq norm(A[:, i]-c) 0.
        count += 1
    end
    @test count == length(collect(1:5:2))

    # rows all
    for (i, r) in enumerate(rows(A))
        @test_approx_eq norm(A[i, :]-r) 0.
        @test typeof(r) == Array{Float64, 2}
    end

    true
end

end

# ----------------------------------------------------------------------------

if length(ARGS) == 1 && first(ARGS) == "test"
    println(Utils.test())
end
