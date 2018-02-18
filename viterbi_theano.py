# viterbi algorithm using theano
import numpy as np
import theano
import theano.tensor as T

obs = {'walk': 0, 'shop': 1, 'clean': 2}
start_prob = np.array([0.4, 0.6])
transition_prob = np.array([[0.6, 0.4], [0.3, 0.7]])
emission_prob = np.array([[0.6, 0.3, 0.1], [0.1, 0.4, 0.5]])
obs_state = [obs['walk'], obs['shop'], obs['clean']]

x = T.ivector('x')
one_x = T.iscalar('one_x')
init = T.vector('init')
transition = T.matrix('transition')
emission = T.matrix('emission')

# get probability of first day
func_init = theano.function(inputs=[one_x, init, emission],
                            outputs=init * emission[:, one_x])
prior_prob = func_init(obs_state[0], start_prob, emission_prob)


def viterbi(seq, prior, transition, emission):
    prior = prior.dimshuffle(0, 'x')
    return T.max(prior * transition * emission[:, seq], axis=0)


result, _ = theano.scan(fn=viterbi,
                        sequences=x,
                        outputs_info=prior_prob,
                        non_sequences=[transition, emission])

scan = theano.function(inputs=[x, transition, emission], outputs=result[-1])
final_prob = scan(obs_state[1:], transition_prob, emission_prob)
print np.max(final_prob)
