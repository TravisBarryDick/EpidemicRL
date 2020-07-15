import PyCall.PyObject

"""
Wrap a python agent so that it supports the julia agent interface.
"""
struct PythonAgent
    pyagt::PyObject
end

copy(agt::PythonAgent) = PythonAgent(agt.pyagt.copy())

start(agt::PythonAgent, state) = agt.pyagt.start(state)

step(agt::PythonAgent, reward, state) = agt.pyagt.step(reward, state)

terminate(agt::PythonAgent, reward) = agt.pyagt.terminate(reward)

"""
Wrap a python environment so that it supports the julia environment
interface.
"""
struct PythonEnvironment
    pyenv::PyObject
end

copy(env::PythonEnvironment) = PythonEnvironment(env.pyenv.copy())

start(env::PythonEnvironment) = env.pyenv.start()

step(env::PythonEnvironment, action) = env.pyenv.step(action)

terminal(env::PythonEnvironment) = env.pyenv.terminal()


"""
Wrap a python policy so that it supports the julia policy interface.
"""
struct PythonPolicy
    pypolicy::PyObject
end

num_params(π::PythonPolicy) = π.num_params()

action_prob(π::PythonPolicy, params, s, a) = π.action_prob(params, s, a)

sample_action(π::PythonPolicy, params, s) = π.sample_action(params, s)

eligibility_vector(π::PythonPolicy, params, s, a) =
    π.eligibility_vector(params, s, a)


"""
Wrap a python feature extractor so that it supporst the julia feature
extractor interface.
"""
struct PythonFeatureExtractor
    py_feature_extractor::PyObject
end

num_features(Φ::PythonFeatureExtractor) = Φ.py_feature_extractor.num_features()

get_features(Φ::PythonFeatureExtractor, s) =
    Φ.py_feature_extractor.get_features(s)
