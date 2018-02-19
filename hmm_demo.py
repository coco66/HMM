import numpy as np
import scipy
from scipy.special import logsumexp
import matplotlib.pyplot as plt 
import pdb


class HMM():
    def __init__(self, n_states, n_obs, Pi=None, A=None, B=None):
        self.n_states = n_states # N
        self.n_obs = n_obs # M
        if Pi:
            self.Pi = Pi
        else:
            self.Pi = np.random.rand(n_states)
            self.Pi = self.Pi / np.sum(self.Pi)

        self.A = np.random.rand(n_states, n_states) # transition probability
        self.A = np.divide(self.A, np.sum(self.A, axis=1, keepdims=True)) 
        self.B = np.random.rand(n_states, n_obs) # observation probability (state -> row, obs -> col)
        self.B = np.divide(self.B, np.sum(self.B, axis=1, keepdims=True))


    def forward(self, obs_sequence):
        alpha = [self.Pi*self.B[:, obs_sequence[0]]] # (n_states,)
        for obs in obs_sequence[1:]:
            alpha.append(np.array([np.dot(alpha[-1],self.A[:,j])*self.B[j,obs] for j in xrange(self.n_states)]))
        return np.array(alpha)

    def log_forward(self, obs_sequence):
        log_alpha = [np.log(self.Pi) + np.log(self.B[:, obs_sequence[0]])] # (n_states,)
        for obs in obs_sequence[1:]:
            log_alpha.append(np.array([logsumexp(log_alpha[-1]+np.log(self.A[:,j])) + np.log(self.B[j,obs]) for j in xrange(self.n_states)]))

        return np.array(log_alpha)


    def backward(self,obs_sequence):
        beta = [np.ones((self.n_states,), dtype=np.float32)]
        for obs in reversed(obs_sequence):
            beta.append([np.dot(self.A[i,:]*self.B[:,obs],beta[-1]) for i in xrange(self.n_states)])
        beta.reverse()
        return np.array(beta)

    def log_backward(self,obs_sequence):
        log_beta = [np.zeros((self.n_states,),dtype=np.float32)]
        for obs in reversed(obs_sequence[1:]):
            log_beta.append([ logsumexp(np.log(self.A[i,:])+np.log(self.B[:,obs])+log_beta[-1]) for i in xrange(self.n_states)])
        log_beta.reverse()
        return np.array(log_beta)


    def baum_welch(self, obs_sequence_list, max_iter=100):
        # number of observation sequences
        K = len(obs_sequence_list)
        log_prob_history = [-5000.0]
        i = 0
        diff = 10
        while((i<max_iter) and (diff > 0.01)):
            log_gammas_list = []
            log_xis_list = []

            pi_num = np.zeros((K, self.n_states))

            A_num = np.zeros((self.n_states, self.n_states, K))
            A_den = np.zeros((K, self.n_states))

            B_den = np.zeros((K, self.n_states))
            B_num = np.zeros((self.n_states, self.n_obs, K))


            # for printing progress
            log_prob = 0


            for k in range(K):
                obs = obs_sequence_list[k]
                T = len(obs)

                log_alphas = self.log_forward(obs).T
                log_betas = self.log_backward(obs).T

                # for printing progress
                last_log_alphas = log_alphas[:, -1]
                log_prob += logsumexp(last_log_alphas)

                log_A = np.log(self.A)
                log_B = np.log(self.B)

                # log_xis is (T-1) by N by N
                log_xis = log_A + \
                    np.expand_dims(log_alphas[:, :-1], axis=0).T + \
                    np.expand_dims(log_betas[:, 1:], axis=1).T + \
                    np.expand_dims(log_B[:, obs[1:]], axis=1).T

                normalization = logsumexp(log_xis, axis=(1, 2))
                log_xis = log_xis - np.expand_dims(np.expand_dims(normalization, axis=0), axis=0).T

                # log_gammas is T by N
                # log_gammas2 = logsumexp(log_xis, axis=2) # can be used to check if xis was calculated correctly
                log_gammas = (log_alphas + log_betas).T
                log_gammas = log_gammas - np.array([logsumexp(log_gammas, axis=1)]).T

                pi_num[k, :] = log_gammas[0, :]

                A_num[:, :, k] = logsumexp(log_xis, axis=0)
                A_den[k, :] = logsumexp(log_gammas[:-1, :], axis=0)

                B_den[k, :] = logsumexp(log_gammas, axis=0)

                for obs_idx in range(self.n_obs):
                    mask = (obs==obs_idx)
                    if np.sum(mask) == 0:
                        B_num[:, obs_idx, k] = -np.inf
                    else:
                        B_num[:, obs_idx, k] = logsumexp(log_gammas[obs==obs_idx, :], axis=0)


            print("log prob before iteration " + str(i) + ": " + str(log_prob))
            log_prob_history.append(log_prob)

            self.Pi = np.exp(logsumexp(pi_num, axis=0) - np.log(K))
            self.A = np.exp(logsumexp(A_num, axis=2) - np.array([logsumexp(A_den, axis=0)]).T)
            self.B = np.exp(logsumexp(B_num, axis=2) - np.array([logsumexp(B_den, axis=0)]).T)
            i += 1
            diff = np.abs(log_prob_history[-1] - log_prob_history[-2])
        return log_prob_history[1:]


    def get_log_prob(self, obs):
        log_alphas = self.log_forward(obs).T
        last_log_alphas = log_alphas[:, -1]
        log_prob = logsumexp(last_log_alphas)
        return log_prob

    def visualize_params(self):
    	fA, axA = plt.subplots()
    	axA.imshow(self.A, cmap='gray') 
    	axA.title('A, transition probability')
    	fB, axB = plt.subplots()
    	axB.imshow(self.B, cmap='gray')
    	axB.title('B, observation probability')
    	plt.show()


def generate_observations(model_name, T):
    """
    The Ride model from west Philly to Engineering.
    State : Chesnut St., Walnut St., Spruce St., Pine St. 
    Observation : Students (five - S, W, P, W, C) 
    model_name : name of a model
    T : length of a observation sequence to generate
    """
    if model_name == 'oober':
        A = np.array([[0.4, 0.4, 0.1, 0.1],
                        [0.3, 0.3, 0.3, 0.1],
                        [0.1, 0.3, 0.3, 0.3],
                        [0.1, 0.1, 0.4, 0.4]], dtype=np.float32)

        B = np.array([[0.0, 0.2, 0.0, 0.2, 0.6],
                        [0.1, 0.4, 0.0, 0.4, 0.1],
                        [0.5, 0.2, 0.1, 0.2, 0.0],
                        [0.3, 0.1, 0.5, 0.1, 0.0]], dtype=np.float32)

        Pi = np.array([0.3, 0.4, 0.1, 0.2], dtype=np.float32)

    elif model_name == 'nowaymo':
        A = np.array([[0.5, 0.1, 0.1, 0.3],
                        [0.2, 0.6, 0.1, 0.1],
                        [0.05, 0.1, 0.8, 0.05],
                        [0, 0.1, 0.2, 0.7]], dtype=np.float32)

        B = np.array([[0.0, 0.2, 0.0, 0.2, 0.6],
                        [0.1, 0.4, 0.0, 0.4, 0.1],
                        [0.5, 0.2, 0.1, 0.2, 0.0],
                        [0.3, 0.1, 0.5, 0.1, 0.0]], dtype=np.float32)

        Pi = np.array([0.2, 0.2, 0.1, 0.5], dtype=np.float32)

    elif model_name == 'dummy':
        A = np.array([[0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        B = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

        Pi = np.array([0.25,0.25,0.25,0.25], dtype=np.float32)

    state = inv_sampling(Pi)
    obs_sequence = []
    for t in xrange(T):
        obs_sequence.append(inv_sampling(B[state,:]))
        state = inv_sampling(A[state,:])
    return np.array(obs_sequence)


def inv_sampling(pdf):
    r = np.random.rand() 
    for (i,p) in enumerate(np.cumsum(pdf)):
        if r <= p:
            return i



if __name__ == '__main__':
    np.random.seed(0)

    T = 100
    oober_data = []
    nowaymo_data = []
    for i in range(10):
        oober_data.append(generate_observations('oober', T))
        nowaymo_data.append(generate_observations('nowaymo', T))


    oober_hmm = HMM(4, 5)
    oober_history = oober_hmm.baum_welch(oober_data, max_iter=30)

    nowaymo_hmm = HMM(4, 5)
    nowaymo_history = nowaymo_hmm.baum_welch(nowaymo_data, max_iter=30)

    pdb.set_trace()

    print("OOBER:\n")
    for i in range(5):
        data = generate_observations('oober', T)
        oober_prob = oober_hmm.get_log_prob(data)
        nowaymo_prob = nowaymo_hmm.get_log_prob(data)
        print("oober: " + str(oober_prob) + "\tnowaymo: " + str(nowaymo_prob))

    print("=============\n")
    print("NOWAYMO:\n")
    for i in range(5):
        data = generate_observations('nowaymo', T)
        oober_prob = oober_hmm.get_log_prob(data)
        nowaymo_prob = nowaymo_hmm.get_log_prob(data)
        print("oober: " + str(oober_prob) + "\tnowaymo: " + str(nowaymo_prob))

