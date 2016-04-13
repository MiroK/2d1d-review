include("utils.jl")
import Utils

# H^s norm operator
snorm_op(s, eigw, eigv, M) = Symmetric((M*eigv)*Diagonal(eigw.^s)*((M*eigv)'))

# Inverse of H^s norm operator
snorm_op_inv(s, eigw, eigv) = Symmetric(eigv*Diagonal(eigw.^(-s))*eigv')

# Approximation of H^s norm operator by only considering it on the subspace
# spanned by upto neigs eigenvectors
function approx_op(neigs, s, eigw, eigv, M)
    @assert 1 <= neigs <= length(eigw)
    eigw = eigw[1:neigs]
    eigv = eigv[:, 1:neigs]
    snorm_op(s, eigw, eigv, M)
end

# The approximation does NOT have an inverse. Pseudoinverse can be defined for
# vertors in the appropriate eigenspace
function approx_op_inv(neigs, s, eigw, eigv)
    @assert 1 <= neigs <= length(eigw)
    eigw = eigw[1:neigs]
    eigv = eigv[:, 1:neigs]
    snorm_op_inv(s, eigw, eigv)
end

# For increasing eigenspaces compute the error in approximation of H^s op
# assembled from n x n matrices
function measure_approx(n, s, op, start=1, update=v-> v*2)
    @assert op in (:norm, :norm_inverse)
    
    println("Measuring approx of $(op) operator")

    A, M = Utils.matrices(n, :Symmetric)
    eigw, eigv = eig(A, M)

    # 'Exact'
    H0 = (op == :norm_inverse) ? snorm_op_inv(s, eigw, eigv) : snorm_op(s, eigw, eigv, M)
    
    neigs = start
    while neigs <= length(eigw)
        H = (op == :norm_inverse) ? approx_op_inv(neigs, s, eigw, eigv) : approx_op(neigs, s, eigw, eigv, M)

        error = norm(H-H0)
        println("$(n) $(neigs) $(error)")

        neigs = update(neigs)
    end

    # Final complete approx
    neigs = length(eigw)
    H = (op == :norm_inverse) ? approx_op_inv(neigs, s, eigw, eigv) : approx_op(neigs, s, eigw, eigv, M)
    error = norm(H-H0)

    println("$(n) $(neigs) $(error)")
end

# ----------------------------------------------------------------------------

# function test()
#     n = 10
#     A, M = Review.matrices(n, :Symmetric)
#     eigw, eigv = eig(A, M)
# 
#     @assert iseye(eigv'*M*eigv)
#     @assert iszeros(snorm_op(1., eigw, eigv, M)-A)
#     @assert iszeros(snorm_op(0., eigw, eigv, M)-M)
#     @assert iseye(snorm_op(0.5, eigw, eigv, M)*snorm_op_inv(0.5, eigw, eigv))
#     @assert iseye(snorm_op_inv(0.5, eigw, eigv)*snorm_op(0.5, eigw, eigv, M))
# 
#     s = 0.4
#     Hs = snorm_op(s, eigw, eigv, M)
#     Hsi = snorm_op_inv(s, eigw, eigv)
#     aHsi = approx_op_inv(length(eigw), s, eigw, eigv)
#     @assert iszeros(Hsi - aHsi)
# 
#     # Operator action
#     r = rand(size(Hs, 1))
#     x0 = Hs*r
# 
#     V = M*eigv
#     x = zeros(x0)
#     x = reduce(+, x, Vector{Float64}[V[:, i]*eigw[i]^s*sum(V[:, i].*r) for i in 1:length(eigw)])
# 
#     @assert norm(x-x0) < 1E-13
# 
# end

#test()

measure_approx(2^3, 0.5, :norm)
