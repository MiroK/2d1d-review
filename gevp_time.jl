include("utils.jl")
using Utils
using PyCall
# Put current dir on path
unshift!(PyVector(pyimport("sys")["path"]), "")

global matrices 
try
    @assert 1 > 5
    # Let there be fenics
    @pyimport gevp_time as fenics
    matrices = fenics.get_1d_matrices
catch
    function matrices(mesh, N, root=".")
        path = string(symbol(mesh, "_", string(N)))
        path = joinpath(root, path)
        mats = readdlm(path)
        A = SymTridiagonal(mats[:, 1], mats[1:end-1, 2])
        M = SymTridiagonal(mats[:, 3], mats[1:end-1, 4])
        A, M
    end
end

lumped = false
mesh, Nrange = "uniform", 1:1
# mesh, Nrange = "nonuniform", 0:6

times = []
sizes = []

for N in Nrange
    mats = matrices(mesh, N)
    # Individual timings
    ts = []

    # Convert to symmetric
    # FIXME when we know what eigenproblem to solve and how
    A, M = map(Symmetric, map(full, mats))
    # Solving the eigenvalue problem
    # FIXME tridiagonality is ignored
    if lumped
        # Do not take M to other side
        tic()
        # Include transformation into timing
        Ml = Symmetric(full(Utils.lumped(M)))
        eigw, eigv = eig(A, Ml)
        push!(ts, toq())

        # Do take M to the other side. This is actually slower than the above
        # tic()
        # Include transformation into timing
        # Mlinv = Utils.lumped(M, true)  # inverse
        # A = full(A)
        # A = Mlinv*A
        # eigw, eigv = eig(A)
        # push!(ts, toq())
    else
        # Convert to symmetric
        tic()
        eigw, eigv = eig(A, M)
        push!(ts, toq())
    end

    @assert all(eigw .> 0)

    # Assembling the preconditioner
    tic()
    H = eigv*Diagonal(eigw.^(-0.5))*eigv'
    push!(ts, toq())
    # Action of preconditioner(matrix)
    x = rand(size(H, 1))
    tic()
    y = H*x
    push!(ts, toq())
    
    #All
    push!(times, ts)
    # Size
    push!(sizes, size(A, 1))


    println("$(sizes[end]) $(times[end]) $(minimum(eigw)) $(maximum(eigw))")
end

record = zeros((length(sizes), 4))
record[:, 1] = sizes
for (row, t) in enumerate(times)
    record[row, 2:end] = t
end

println(record)
