using StatsBase

# Distance function -------------------------------------------------

"""
    euclidean(a::AbstractVector, b::AbstractVector)

Calculate euclidean distance from two vectors. √∑(aᵢ - bᵢ)²
"""
function euclidean(a::AbstractVector{T},
                   b::AbstractVector{T}) where {T<:AbstractFloat}
    @assert length(a) == length(b)

    # euclidean(a, b) = √∑(aᵢ- bᵢ)²
    s = zero(T)
    @simd for i in eachindex(a)
        @inbounds s += (a[i] - b[i])^2
    end
    return √s
end

# KnnClassifier -----------------------------------------------------
struct KnnClassifier{T<:Real, S}
    X::Matrix{T}
    y::Vector{S}
    K::Int

    # Internal constructor
    function KnnClassifier(X::Matrix{T}, y::Vector{S}, K::Int) where {T<:Real, S}
        @assert size(X, 1) == size(y, 1)
        return new{T, S}(X, y, K)
    end
end

# External constructor
function KnnClassifier(Xv::Vector{T}, y::Vector{S}, K::Int) where {T<:Real, S}
    X = reshape(Xv, length(Xv), 1)
    return KnnClassifier(X, y, K)
end

function predict(model::KnnClassifier{T, S}, input::Vector{T}) where {T<:Real, S}
    nl, nc = size(model.X, 1), size(model.X, 2)

    @assert size(input, 1) == nc

    # calculate all the distances
    dists = Vector{T}(undef, nl)
    for i in axes(model.X, 1)
        dists[i] = euclidean(input, model.X[i, :])
    end

    # sort index of distances and select the k nearest indexes
    idx = sortperm(dists)[1:model.K]

    # select the KNearestNeighbours
    kneighbours = model.y[idx]

    # get prediction
    return mode(kneighbours)
end

function predict(model::KnnClassifier{T, S}, input::Matrix{T}) where {T<:Real, S}
    nl, nc = size(model.X, 1), size(model.X, 2)
    nli = size(input, 1)

    @assert size(input, 2) == nc

    results = Vector{S}(undef, nli)
    for j in axes(input, 1)

        # calculate all the distances
        dists = Vector{T}(undef, nl)
        for i in axes(model.X, 1)
            dists[i] = euclidean(input[j, :], model.X[i, :])
        end
    
        # sort index of distances and select the k nearest indexes
        idx = sortperm(dists)[1:model.K]
    
        # select the KNearestNeighbours
        kneighbours = model.y[idx]
        
        # get prediction
        results[j] = mode(kneighbours)
    end
    
    return results
end


# KnnRegression -----------------------------------------------------
struct KnnRegression{T<:Real, S<:Real}
    X::Matrix{T}
    y::Vector{S}
    K::Int

    # Internal constructor
    function KnnRegression(X::Matrix{T}, y::Vector{S}, K::Int) where {T<:Real, S<:Real}
        @assert size(X, 1) == size(y, 1)
        return new{T, S}(X, y, K)
    end
end

# External constructor
function KnnRegression(Xv::Vector{T}, y::Vector{S}, K::Int) where {T<:Real, S<:Real}
    X = reshape(Xv, length(Xv), 1)
    return new{T, S}(X, y, K)
end

function predict(model::KnnRegression{T, S}, input::Vector{T}, w::Bool=false) where {T<:Real, S<:Real}
    nl, nc = size(model.X, 1), size(model.X, 2)

    @assert size(input, 1) == nc

    # calculate all the distances
    dists = Vector{T}(undef, nl)
    for i in axes(model.X, 1)
        dists[i] = euclidean(input, model.X[i, :])
    end

    # sort index of distances and select the k nearest indexes
    idx = sortperm(dists)[1:model.K]

    # select the KNearestNeighbours
    y_kneighbours = model.y[idx]

    # get prediction by mean or weighted mean
    if w
        k_min_dists = sort(dists)[1:model.K]
        return mean(y_kneighbours, weights(k_min_dists))
    else
        return mean(y_kneighbours)
    end

end

function predict(model::KnnRegression{T, S}, input::Matrix{T}, w::Bool=false) where {T<:Real, S<:Real}
    nl, nc = size(model.X, 1), size(model.X, 2)
    nli = size(input, 1)

    @assert size(input, 2) == nc

    results = Vector{S}(undef, nli)
    for j in axes(input, 1)

        # calculate all the distances
        dists = Vector{T}(undef, nl)
        for i in axes(model.X, 1)
            dists[i] = euclidean(input[j, :], model.X[i, :])
        end

        # sort index of distances and select the k nearest indexes
        idx = sortperm(dists)[1:model.K]

        # select the KNearestNeighbours
        y_kneighbours = model.y[idx]

        # get prediction by mean or weighted mean
        if w
            k_min_dists = sort(dists)[1:model.K]
            results[j] = mean(y_kneighbours, weights(k_min_dists))
        else
            results[j] = mean(y_kneighbours)
        end
    end

    return results
end