""" 
A NonLearningAgent uses a parameterized policy but does not do any
learning (i.e., it never updates the policy parameters).
"""
struct NonLearningAgent{Policy}
    π::Policy
    θ::Vector{Float64}
end

copy(agt::NonLearningAgent) = agt

start(agt::NonLearningAgent, state) = sample_action(agt.π, agt.θ, state)

step(agt::NonLearningAgent, reward, state) = sample_action(agt.π, agt.θ, state)

terminate(agt::NonLearningAgent, reward) = nothing
