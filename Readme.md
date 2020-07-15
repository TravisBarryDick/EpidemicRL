# EpidemicRL

This package contains reinforcement learning agents and environments
for epidemiological models. 

## Interfaces:

The EpidemicRL package makes use of the following informal interfaces. 

### Basic Interfaces

#### Environment Interface

Environments must implement the following functions:

- `copy(env)` creates a copy of the environment.
- `start(env)` starts an episode in the environment and returns the
  first state.
- `step(env, action)` takes a step in the environment's current
  episode producing a `reward` and transitioning to the next
  `state`. The reward and next state are returned as a pair `(reward,
  state)`.
- `terminal(env)` tests if the episode is in a terminal state.

#### Agent Interface

Agents must implement the following functions:

- `copy(env)` creates a copy of the agent.
- `start(agt, state)` starts an episode for the agent and returns the
  action it chooses to take from the given `state`.
- `step(agt, reward, state)` informs the agent of the `reward`
  produced by their last action and new `state` of the
  environment. Return the next action to take.
- `terminate(agt, reward)` informs the agent that the epsiode has
  ended and gives them the final reward.

### Policy Gradient Interfaces

#### Policy Interface

Parametric policies must implement the following functions:

- `num_params(param_policy)` returns the number of parameters for the
  policy.
- `action_prob(param_policy, params, s, a)` returns the probability of
  choosing action `a` in state `s` with parameters `params` for the
  policy `param_policy`. If the action space is continuous, this
  function should return value of the action probability density
  function.
- `sample_action(param_policy, params, s)` samples an action for state
  `s` with parameters `params` for the policy `param_policy`.
- `eligibility_vector(param_policy, params, s, a)` returns the
  "eligibility vector" for the state-action pair `(s,a)` for
  parameterized policy `param_policy` with parameters `params`. The
  eligibility vector is the gradient of `theta ->
  log(action_prob(param_policy, theta, s, a))` with respect to `theta`
  and evaluated at `params`.
  
  - Note: there is a default implementation of `eligibility_vector`
    that uses Zygote automatic differentiation, so implementation is
    optional (however, hand-implemented eligibility vector code will
    likely be significantly faster).

#### FeatureExtractor Interface

Feature extractors must implement the following functions:

- `num_features(feature_extractor)` returns the number of features
  output by the feature extractor.
- `get_features(feature_extractor, s)` returns the feature vector for
  state `s`.

