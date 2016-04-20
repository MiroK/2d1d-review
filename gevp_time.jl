include("utils.jl")
using Utils
    

root = "./jl_matrices"
mesh = "nonuniform"
record = []

for f in sort!(filter(x->startswith(x, mesh), readdir(root)))
    # Load matrix from file
    path = joinpath(root, f)
    mats = Matrix{Float64}(readdlm(path))
    A = SymTridiagonal(mats[:, 1], mats[1:end-1, 2])
    M = SymTridiagonal(mats[:, 3], mats[1:end-1, 4])

    row = Real[size(A, 1)]

    # Time lumped problem
    tic()
    Minv = Utils.lumped(M, -0.5)
    A = ⋆(A, Minv)
    @assert typeof(A) == SymTridiagonal{eltype(A)}
    eigw, eigv = eig(A)
    push!(row, toq())
    @assert all(eigw .> 0)

    # # Assembling the preconditioner
    tic()
    H = eigv*Diagonal(eigw.^(-0.5))*eigv'
    push!(row, toq())

    # # Action of preconditioner(matrix)
    x = rand(size(H, 1))
    tic()
    y = H*x
    push!(row, toq())

    # # For fum timing of unlumped solver
    A = SymTridiagonal(mats[:, 1], mats[1:end-1, 2])
    M = SymTridiagonal(mats[:, 3], mats[1:end-1, 4])
    A, M = Symmetric(full(A)), Symmetric(full(M))
    tic()
    eigw0, eigv0 = eig(A, M)
    push!(row, toq())

    # # Compare the spectram if     inv(lump(M))*A vs inv(M)*A
    c = minimum(eigw0)/minimum(eigw)
    C = maximum(eigw0)/maximum(eigw)
    push!(row, c, C)

    push!(record, row)

    println(record[end])
end

nrows, ncols = length(record), length(first(record))
data = zeros(Real, nrows, ncols)
for (r, row) in enumerate(sort!(record, by=first))
    data[r, :] = row
end

writedlm(open(joinpath("./data", "jl_"*mesh), "w"), data)
