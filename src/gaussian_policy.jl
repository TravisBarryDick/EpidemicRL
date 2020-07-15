import Zygote.dropgrad
import LinearAlgebra: norm, dot

"Feature extractor that always returns [1.0]"
struct ConstantFeatureExtractor end
num_features(Φ::ConstantFeatureExtractor) = 1
get_features(Φ::ConstantFeatureExtractor, s) = [1.0]


""" 
Represents a policy over real-valued actions where the action
distribution is Gaussian with mean and standard deviation depending on
the current state.

The action distribution for state `s` is computed as follows:

1. The FeatureExtractor `Φ` is used to compute a feature vector `φ`
for state `s`.
2. The mean is computed as `μ = dot(θ_μ, φ)`, where θ_μ are the mean
parameters.
3. The std deviation is computed as `σ = slf(σ_min, σ_max, dot(θ_σ,
   φ))` where `slf` is a shifted logistic function that softly clamps
   its third argument to be in the range [σ_min, σ_max] and θ_σ are
   the std deviation parameters.
4. The action distribution is `Normal(μ, σ)`.

The single parameter vector for this policy is the concatenation of
`θ_μ` and `θ_σ`: `θ = [θ_μ ; θ_σ]`.

"""
struct GaussianPolicy{FeatureExtractor}
    Φ::FeatureExtractor
    σ_min::Float64
    σ_max::Float64
end

num_params(π::GaussianPolicy) = 2 * num_features(π.Φ)

function get_μ(π::GaussianPolicy, θ, φ)
    d = length(φ)
    dot(θ[1:d], φ)
end

function shifted_logistic(min_value, max_value, x)
    min_value + (max_value - min_value) / 2 * (1 + tanh(x / 2))
end

function get_σ(π::GaussianPolicy, θ, φ)
    d = length(φ)
    score = dot(θ[d+1:2d], φ)
    shifted_logistic(π.σ_min, π.σ_max, score)
end

function action_prob(π::GaussianPolicy, θ, s, a)
    φ = dropgrad(get_features(π.Φ, s))
    μ = get_μ(π, θ, φ)
    σ = get_σ(π, θ, φ)
    1 / (σ * sqrt(2 * pi)) * exp(-((a - μ) / σ)^2 / 2)
end

function sample_action(π::GaussianPolicy, θ, s)
    φ = get_features(π.Φ, s)
    μ = get_μ(π, θ, φ)
    σ = get_σ(π, θ, φ)
    return randn() * σ + μ
end

function eligibility_vector(π::GaussianPolicy, θ, s, a)
    φ = get_features(π.Φ, s)
    d = length(φ)
    μ = get_μ(π, θ, φ)
    σ = get_σ(π, θ, φ)
    ev = similar(θ)
    ev[1:d] = (a - μ) / σ^2 * φ
    ev[d+1:2d] =
        1 / σ *
        (((a - μ) / σ)^2 - 1) *
        (σ - π.σ_min) *
        (1 - (σ - π.σ_min) / (π.σ_max - π.σ_min)) *
        φ
    ev
end

"""
Represents a policy over real-valued actions that ignores the current
state and always outputs a sample drawn from a Gaussian
distribution. The standard deviation of the distribution is fixed when
the `SimpleGaussianPolicy` is constructed, while the mean of the
distribution is the single parameter of the policy.
"""
struct SimpleGaussianPolicy
    σ::Float64
end

num_params(π::SimpleGaussianPolicy) = 1

function action_prob(π::SimpleGaussianPolicy, θ)
    σ = π.σ
    μ = θ[1]
    1 / (σ * sqrt(2 * pi)) * exp(-((a - μ) / σ)^2 / 2)
end

function sample_action(π::SimpleGaussianPolicy, θ)
    σ = π.σ
    μ = θ[1]
    rand() * σ + μ
end
