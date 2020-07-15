using ProgressMeter

"""
    do_episode(env, agt)

Simulate an episode between the environment `env` and agent `agt`. A
copy of the environment is used to avoid mutation. The agent is not
copied so that any mutations are persisted. Returns vectors of the
sequence of states, actions, and rewards encountered during the
episode.
"""
function do_episode(env, agt)
    env = copy(env)

    state = start(env)
    action = start(agt, state)

    states = [state]
    actions = [action]
    rewards = Float64[]

    while !terminal(env)
        reward, state = step(env, action)
        action = step(agt, reward, state)

        push!(states, state)
        push!(actions, action)
        push!(rewards, reward)
    end

    terminate(agt, rewards[end])

    return states, actions, rewards
end

"""
    do_episode_return(env, agt)

Identical to `do_episode` but returns the total reward (i.e., the
return) for the episode.
"""
do_episode_return(env, agt) = sum(do_episode(env, agt)[3])

"""
    do_episodes(env, agt, num_episodes)

Simulate `num_episodes` episodes between the environment `env` and
agent `agt`. The environment is copied for each episode but the agent
is not, so updates to the agent are persisted. Returns a vector of the
agent's return on each episode.
"""
function do_episodes(env, agt, num_episodes; verbose = false)
    returns = Vector{Float64}(undef, num_episodes)
    wait_time = verbose ? 1 : Inf
    @showprogress wait_time for episode = 1:num_episodes
        states, actions, rewards = do_episode(env, agt)
        returns[episode] = sum(rewards)
    end
    return returns
end

"""
    learning_curve(env, agt, num_episodes, num_runs)

"""
function learning_curve(env, agt, num_episodes, num_runs; verbose = false)
    lc = zeros(num_episodes)
    wait_time = verbose ? 1 : Inf
    p = Progress(num_episodes * num_runs, wait_time)
    for run = 1:num_runs
        run_agt = copy(agt)
        for episode = 1:num_episodes
            states, actions, rewards = do_episode(env, run_agt)
            lc[episode] += sum(rewards) / num_runs
            next!(p)
        end
    end
    return lc
end

"""
    returns(rewards, [discount = 1.0])

Compute the discounted return on each step of an episode given the
vector of rewards encountered on that trajectory. 
"""
function get_returns(rewards, discount = 1.0)
    returns = Vector{Float64}(undef, length(rewards))
    returns[end] = rewards[end]
    for i = length(rewards)-1:-1:1
        returns[i] = discount * returns[i+1] + rewards[i]
    end
    return returns
end
