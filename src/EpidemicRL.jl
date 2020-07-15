
module EpidemicRL

# ----- Base Overrides ----- #

import Base: copy

# ----- Informal Interface Methods ----- #

# Agents and Environments
export start, step, terminal, terminate

# Policy Gradient
export num_params, action_prob, sample_action, eligibility_vector
export num_features, get_features

# ----- Implementation Exports ----- #

include("sir_simulation.jl")
export SIRQState, population_size, SIRQEnvironment

include("fraction_agent.jl")
export FractionAgent, LogitNormalFractionPolicy

include("action_transformer.jl")
export ActionTransformer

include("optimizers.jl")
export get_params, update
export GradientAscent, Adam

include("policy_gradient.jl")
export ReinforceAgent

include("non_learning_agent.jl")
export NonLearningAgent

include("gaussian_policy.jl")
export GaussianPolicy, ConstantFeatureExtractor, SimpleGaussianPolicy

include("python_wrappers.jl")
export PythonAgent, PythonEnvironment, PythonPolicy, PythonFeatureExtractor

include("utils.jl")
export do_episode, do_episodes, learning_curve, get_returns

end
