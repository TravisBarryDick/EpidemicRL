struct SIRQState
    S::Float64      # Number of susceptible individuals
    I::Float64      # Number of infected individuals
    R::Float64      # Number of removed individuals
    Q::Float64      # Number of quarantines / isolations remaining
    max_I::Float64  # Maximum number of infections encountered so far
end

SIRQState(S, I, R, Q) = SIRQState(S, I, R, Q, 0)

population_size(s::SIRQState) = s.S + s.I + s.R

mutable struct SIRQEnvironment
    state::SIRQState
    beta::Float64
    gamma::Float64
end

import Base.copy
copy(env::SIRQEnvironment) = SIRQEnvironment(env.state, env.beta, env.gamma)

start(env::SIRQEnvironment) = env.state

function step(env::SIRQEnvironment, num_quarantines)
    num_quarantines = max(0, min(num_quarantines, env.state.Q, env.state.I))
    S_to_I = min(
        env.state.S,
        env.beta * env.state.S * env.state.I / population_size(env.state),
    )
    I_to_R = min(env.state.I, env.gamma * env.state.I + num_quarantines)

    next_S = env.state.S - S_to_I
    next_I = env.state.I + S_to_I - I_to_R
    next_R = env.state.R + I_to_R
    next_Q = env.state.Q - num_quarantines
    next_max_I = max(env.state.max_I, next_I)
    next_state = SIRQState(next_S, next_I, next_R, next_Q, next_max_I)

    reward = 0.0
    if next_max_I > env.state.max_I
        reward = -(next_max_I - env.state.max_I)
    end

    env.state = next_state

    return reward, next_state
end

terminal(env::SIRQEnvironment) = env.state.I < 1e-10
