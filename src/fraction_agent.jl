struct FractionAgent
    fraction::Float64
end

get_action(agt::FractionAgent, state::SIRQState) = agt.fraction * state.I

start(agt::FractionAgent, state) = get_action(agt, state)

step(agt::FractionAgent, reward, state) = get_action(agt, state)

terminate(agt::FractionAgent, reward) = nothing


# ----- LogitNormalFractionPolicy ----- #


logit(x) = log(x / (1 - x))

logistic(x) = 0.5 + 0.5 * tanh(x / 2)

"Density function for a logit normal random variable."
logit_normal_pdf(μ, σ, x) =
    1 / (σ * sqrt(2 * pi)) * 1 / (x * (1 - x)) *
    exp(-(logit(x) - μ)^2 / (2 * σ^2))

struct LogitNormalFractionPolicy
    σ::Float64
end

num_params(π::LogitNormalFractionPolicy) = 1

function action_prob(π::LogitNormalFractionPolicy, θ, s, a)
    σ = π.σ
    μ = θ[1]
    fraction = a / s.I
    logit_normal_pdf(μ, σ, fraction) / s.I
end

function sample_action(π::LogitNormalFractionPolicy, θ, s)
    σ = π.σ
    μ = θ[1]
    logistic(randn() * σ + μ) * s.I
end
