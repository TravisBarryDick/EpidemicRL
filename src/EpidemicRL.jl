module EpidemicRL

# ----- Base Overrides ----- #

import Base: copy

# ----- Informal Interface Methods ----- #

# Agents and Environments
export start, step, terminal, terminate

# Policy Gradient
export num_params, action_prob, sample_action, eligibility_vector
export num_features, get_features

# ----- SIRQ Environment ----- #

include("sir_simulation.jl")
export SIRQState, population_size, SIRQEnvironment, fraction_to_quarantines

include("sirq_tilecoder.jl")
export SIRQTileCoder

# ----- Fraction Agent ----- #

include("fraction_agent.jl")
export FractionAgent

# ----- Policy Gradient Methods ----- #

include("optimizers.jl")
export get_params, update
export GradientAscent, Adam

include("reinforce.jl")
export ReinforceAgent

include("actor_critic.jl")
export ActorCriticAgent, Actor, Critic, LinearValueFn

# ----- Policies ------ #

include("logit_normal_policy.jl")
export LogitNormalPolicy

include("gaussian_policy.jl")
export GaussianPolicy, SimpleGaussianPolicy

# ----- Feature Extractors ----- #

include("constant_feature_extractor.jl")
export ConstantFeatureExtractor

# ----- Utilties ----- #

include("action_transformer.jl")
export ActionTransformer

include("non_learning_agent.jl")
export NonLearningAgent

include("python_wrappers.jl")
export PythonAgent, PythonEnvironment, PythonPolicy, PythonFeatureExtractor

include("utils.jl")
export do_episode, do_episodes, learning_curve, get_returns

end
