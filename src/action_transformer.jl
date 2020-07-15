struct ActionTransformer{A,T}
    agt::A
    transformer::T
end

copy(agt::ActionTransformer) = ActionTransformer(copy(agt.agt), agt.transformer)

start(agt::ActionTransformer, state) =
    agt.transformer(state, start(agt.agt, state))

step(agt::ActionTransformer, reward, state) =
    agt.transformer(state, step(agt.agt, reward, state))

terminate(agt::ActionTransformer, reward) = terminate(agt.agt, reward)
