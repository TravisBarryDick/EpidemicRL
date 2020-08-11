logit(x) = log(x / (1 - x))

logistic(x) = 0.5 + 0.5 * tanh(x / 2)

"Density function for a logit normal random variable."
logit_normal_pdf(μ, σ, x) =
    1 / (σ * sqrt(2 * pi)) * 1 / (x * (1 - x)) *
    exp(-(logit(x) - μ)^2 / (2 * σ^2))

""" 
Represents a policy over real-valued actions where the action
distribution is LogitNormal with μ parameter depending on the current
state and a fixed σ parameter.

The action distribution for state `s` is computed as follows:

1. The FeatureExtractor `Φ` is used to compute a feature vector `φ`
   for state `s`.
2. The μ parameter is computed as `μ = dot(θ_μ, φ)`, where θ is the
   vector of policy parameters.
4. The action distribution is `LogitNormal(μ, σ)`.
"""
struct LogitNormalPolicy{FeatureExtractor}
    Φ::FeatureExtractor
    σ::Float64
end

num_params(π::LogitNormalPolicy) = num_params(π.Φ)

function action_prob(π::LogitNormalPolicy, θ, s, a)
    σ = π.σ
    μ = dot(θ, dropgrad(get_features(π.Φ, s)))
    logit_normal_pdf(μ, σ, a)
end

function sample_action(π::LogitNormalPolicy, θ, s)
    σ = π.σ
    μ = dot(θ, dropgrad(get_features(π.Φ, s)))
    logistic(randn() * σ + μ)
end

function eligibility_vector(π::LogitNormalPolicy, θ, s, a)
    fv = get_features(π.Φ, s)
    μ = dot(fv, θ)
    la = logit(a)
    (la - μ) / la * fv
end
