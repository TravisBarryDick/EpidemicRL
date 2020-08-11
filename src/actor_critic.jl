# ----- Actor ----- #

mutable struct Actor{Policy,Optimizer}
    policy::Policy
    opt::Optimizer
    trace::Vector{Float64}
    trace_decay::Float64
end

Actor(π, opt, λ) = Actor(π, opt, zero(get_params(opt)), λ)

copy(a::Actor) = Actor(a.policy, copy(a.opt), copy(a.trace), a.trace_decay)

start(a::Actor) = fill!(a.trace, 0.0)

sample_action(a::Actor, state) =
    sample_action(a.policy, get_params(a.opt), state)

function update(a::Actor, state, action, δ)
    a.trace *= a.trace_decay
    a.trace += eligibility_vector(a.policy, get_params(a.opt), state, action)
    update(a.opt, δ * a.trace)
end

# ----- Critic ----- #

mutable struct Critic{ValueFn,Optimizer}
    value_fn::ValueFn
    opt::Optimizer
    trace::Vector{Float64}
    trace_decay::Float64
end

Critic(value_fn, opt, λ) = Critic(value_fn, opt, zero(get_params(opt)), λ)

copy(c::Critic) = Critic(c.value_fn, copy(c.opt), copy(c.trace), c.trace_decay)

start(c::Critic) = fill!(c.trace, 0.0)

value(c::Critic, state) = value(c.value_fn, get_params(c.opt), state)

function update(c::Critic, state, δ)
    c.trace *= c.trace_decay
    c.trace += value_gradient(c.value_fn, get_params(c.opt), state)
    update(c.opt, δ * c.trace)
end

# ----- Actor Critic Agent ----- #

"""
A simple container for holding a potentially uninitialized value of
type T. No error checking is done, so be sure that `val` is
initialized before accessing.
"""
struct Maybe{T}
    val::T
    Maybe{T}() where {T} = new{T}()
    Maybe(val::T) where {T} = new{T}(val)
end

mutable struct ActorCriticAgent{Actor,Critic,State,Action}
    actor::Actor
    critic::Critic
    last_s::Maybe{State}
    last_a::Maybe{Action}
end

ActorCriticAgent{S,A}(actor, critic) where {S,A} =
    ActorCriticAgent(actor, critic, Maybe{S}(), Maybe{A}())

copy(agt::ActorCriticAgent) =
    ActorCriticAgent(copy(agt.actor), copy(agt.critic), agt.last_s, agt.last_a)

function start(agt::ActorCriticAgent, state)
    start(agt.actor)
    start(agt.critic)
    agt.last_s = Maybe(state)
    action = sample_action(agt.actor, state)
    agt.last_a = Maybe(action)
    action
end

function step(agt::ActorCriticAgent, reward, state)
    δ = reward + value(agt.critic, state) - value(agt.critic, agt.last_s.val)
    update(agt.critic, agt.last_s.val, δ)
    update(agt.actor, agt.last_s.val, agt.last_a.val, δ)
    action = sample_action(agt.actor, state)
    agt.last_s = Maybe(state)
    agt.last_a = Maybe(action)
    action
end

function terminate(agt::ActorCriticAgent, reward)
    δ = reward - value(agt.critic, agt.last_s.val)
    update(agt.critic, agt.last_s.val, δ)
    update(agt.actor, agt.last_s.val, agt.last_a.val, δ)
    nothing
end

struct LinearValueFn{FeatureExtractor}
    Φ::FeatureExtractor
end

num_params(lc::LinearValueFn) = num_params(lc.Φ)

value(lc::LinearValueFn, θ, s) = dot(θ, dropgrad(get_features(lc.Φ, s)))

value_gradient(lc::LinearValueFn, Θ, s) = get_features(lc.Φ, s)

function value_gradient(c, θ, s)
    ∇ = gradient(θ -> value(c, θ, s), θ)[1]
    if any(isnan.(∇)) || any(isinf.(∇))
        return zero(θ)
    end
    ∇
end
