struct FractionAgent
    fraction::Float64
end

copy(agt::FractionAgent) = agt

get_action(agt::FractionAgent, state::SIRQState) = agt.fraction * state.I

start(agt::FractionAgent, state) = get_action(agt, state)

step(agt::FractionAgent, reward, state) = get_action(agt, state)

terminate(agt::FractionAgent, reward) = nothing

