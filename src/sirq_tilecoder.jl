import SparseArrays

struct SIRQTileCoder
    S_bins::Int
    S_min::Float64
    S_max::Float64
    
    I_bins::Int
    I_min::Float64
    I_max::Float64
    
    R_bins::Int
    R_min::Float64
    R_max::Float64
    
    Q_bins::Int
    Q_min::Float64
    Q_max::Float64
    
    max_I_bins::Int
    max_I_min::Float64
    max_I_max::Float64
end

num_features(Φ::SIRQTileCoder) =
    Φ.S_bins * Φ.I_bins * Φ.R_bins * Φ.Q_bins * Φ.max_I_bins

get_bin(min_val, max_val, nbins, x) =
    clamp(floor(Int, (x - min_val) / (max_val - min_val) * nbins) + 1, 1, nbins)

function get_features(Φ::SIRQTileCoder, s::SIRQState)
    linear_indices = LinearIndices((
        1:Φ.S_bins,
        1:Φ.I_bins,
        1:Φ.R_bins,
        1:Φ.Q_bins,
        1:Φ.max_I_bins,
    ))

    n = population_size(s)
    
    tile_coord = CartesianIndex(
        get_bin(Φ.S_min, Φ.S_max, Φ.S_bins, s.S),
        get_bin(Φ.I_min, Φ.I_max, Φ.I_bins, s.I),
        get_bin(Φ.R_min, Φ.R_max, Φ.R_bins, s.R),
        get_bin(Φ.Q_min, Φ.Q_max, Φ.Q_bins, s.Q),
        get_bin(Φ.max_I_min, Φ.max_I_max, Φ.max_I_bins, s.max_I),
    )
    
    SparseArrays.SparseVector(num_features(Φ), [linear_indices[tile_coord]], [1.0])
end
