# viterbi algorithm
state = ['sunny', 'rainy']
obs = ['walk', 'shop', 'clean']
start_prob = {state[0]: 0.4, state[1]: 0.6}
transition_prob = {state[0]: {state[0]: 0.6, state[1]: 0.4},
                   state[1]: {state[0]: 0.3, state[1]: 0.7}}
emission_prob = {state[0]: {obs[0]: 0.6, obs[1]: 0.3, obs[2]: 0.1},
                 state[1]: {obs[0]: 0.1, obs[1]: 0.4, obs[2]: 0.5}}

obs_state = [obs[0], obs[1], obs[2]]  # walk, shop, clean

hmm = []
dic_ = {}
# initialization
for s in state:
    dic_[s] = emission_prob[s][obs[0]] * start_prob[s]
hmm.append(dic_)
# induction
i = 1
dic_ = {}
for o in obs[i:]:
    for s in state:  # current state
        dic_[s] = max([hmm[i - 1][s_] * transition_prob[s_][s] * emission_prob[s][o] for s_ in state])
    hmm.append(dic_)
    i += 1
print [sorted(state.items(), key=lambda x: x[1], reverse=True)[0][0] for state in hmm]
