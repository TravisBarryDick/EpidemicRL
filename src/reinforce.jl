import Zygote.gradient

mutable struct ReinforceAgent{Policy,State,Action,Optimizer}
    π::Policy
    opt::Optimizer

    states::Vector{State}
    actions::Vector{Action}
    rewards::Vector{Float64}
end

ReinforceAgent{State,Action}(π, opt) where {State,Action} =
    ReinforceAgent(π, opt, State[], Action[], Float64[])

ReinforceAgent(::Type{State}, ::Type{Action}, π, opt) where {State,Action} =
    ReinforceAgent(π, opt, State[], Action[], Float64[])

copy(agt::ReinforceAgent) = ReinforceAgent(
    agt.π,
    copy(agt.opt),
    copy(agt.states),
    copy(agt.actions),
    copy(agt.rewards),
)

function start(agt::ReinforceAgent, state)
    agt.states = [state]
    action = sample_action(agt.π, get_params(agt.opt), state)
    agt.actions = [action]
    agt.rewards = Float64[]
    action
end

function step(agt::ReinforceAgent, reward, state)
    push!(agt.rewards, reward)
    push!(agt.states, state)
    action = sample_action(agt.π, get_params(agt.opt), state)
    push!(agt.actions, action)
    action
end

function terminate(agt::ReinforceAgent, reward)
    push!(agt.rewards, reward)
    T = length(agt.states)
    ∇ = zero(get_params(agt.opt))
    G = 0.0
    θ = get_params(agt.opt)
    for t = T:-1:1
        s = agt.states[t]
        a = agt.actions[t]
        G += agt.rewards[t]
        ∇ += G * eligibility_vector(agt.π, θ, s, a) / T
    end
    update(agt.opt, ∇)
end

function eligibility_vector(π, θ, s, a)
    ev = gradient(θ -> log(action_prob(π, θ, s, a)), θ)[1]
    if any(isnan.(ev)) || any(isinf.(ev))
        return zero(θ)
    end
    ev
end
