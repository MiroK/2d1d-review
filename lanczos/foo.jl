include("../utils.jl")
using Utils
using IterativeSolvers

ncells = 100
A, _ = Utils.matrices(ncells, :SymTridiagonal)

order = size(A, 2)
n = 10
K = KrylovSubspace(A, n, order, eltype(A))
q1 = [1.; zeros(eltype(A), order-1)]
IterativeSolvers.append!(K, q1)
