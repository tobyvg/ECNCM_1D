using LinearAlgebra
using SparseArrays
using Random
using CSV,Tables

function construct_weighted_interpolation_matrix(x1,x2)
    @assert x1[1] <= x2[1] && x1[end] >= x2[end]
    dx1 = x1[2]-x1[1]
    dx2 = x2[2]-x2[1]
    x1_edges = x1 .+ 1/2*dx1
    x2_edges = x2 .+ 1/2*dx2
    x1_edges = [0. ; x1_edges]
    x2_edges = [0. ; x2_edges]
    mat = spzeros(size(x2)[1],size(x1)[1])
    for i in 1:size(x2_edges)[1]-1
        for j in 1:size(x1_edges)[1]-1
            d1 = x2_edges[i] - x1_edges[j+1]
            d2 = x2_edges[i+1] - x1_edges[j]
            if d1 <= 0 && d2 >= 0
                if abs(d1) >=dx1 && abs(d2) >=dx1
                    overlap = dx1
                elseif abs(d1) >= dx1
                    overlap = abs(d2)
                elseif abs(d2) >= dx1
                    overlap = abs(d1)
                else
                    overlap = dx1
                end
            mat[i,j] = overlap/dx2
            end
        end
    end
    return mat
end

function index_converter(i,I)
    index = mod(i,I)
    if index == 0
        index = I
    end
    return index
end

function process_path(path)
    if last(path,1) != "/"
        path = path * "/"
    end
    return path
end



function save_heatmap_data(path,x,y,z)
    path = process_path(path)
    mkpath(path)
    CSV.write(path * "x.csv",Tables.table(x))
    CSV.write(path * "y.csv",Tables.table(y))
    CSV.write(path * "z.csv",Tables.table(z))
    return 0
end

function import_heatmap_data(path)
    path = process_path(path)
    x = CSV.File(path * "x.csv") |> Tables.matrix
    y = CSV.File(path * "y.csv") |> Tables.matrix
    z = CSV.File(path * "z.csv") |> Tables.matrix
    return x[:,1],y[:,1],z
end

function cut_indexes(d,fraction;randomize = false,uniform = true)
    d_size = size(d)[end]
    if uniform != true
        random_vec =  collect(1:d_size)
        random_vec = shuffle(random_vec)
    end
    amount = fraction*d_size
    step_size = floor(Int,d_size/amount)
    indexes = [1]
    index = 1
    while index <= d_size - step_size
        index += step_size
        if uniform
            indexes = [indexes ;index]
        else
            indexes = [indexes ;random_vec[index]]
        end
    end
    if randomize
        indexes = shuffle(indexes)
    end
    return indexes
end



function adj_flatten(x)
    return reshape(x,(prod(size(x)),))
end
