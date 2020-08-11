import SparseArrays

struct SIRQTileCoder
    S_bins::Int
    I_bins::Int
    R_bins::Int
    Q_bins::Int
    max_I_bins::Int
end

num_features(Φ::SIRQTileCoder) =
    Φ.S_bins * Φ.I_bins * Φ.R_bins * Φ.Q_bins * Φ.max_I_bins

get_bin(min_val, max_val, nbins, x) =
    min(floor(Int, (x - min_val) / (max_val - min_val) * nbins) + 1, nbins)

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
        get_bin(0, n, Φ.S_bins, s.S),
        get_bin(0, n, Φ.I_bins, s.I),
        get_bin(0, n, Φ.R_bins, s.R),
        get_bin(0, n, Φ.Q_bins, s.Q),
        get_bin(0, n, Φ.max_I_bins, s.max_I),
    )
    SparseArrays.SparseVector(num_features(Φ), [linear_indices[tile_coord]], [1.0])
end
