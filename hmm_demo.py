import numpy as np
from scipy.special import logsumexp
class HMM():
	def __init__(self,n_states, n_obs, Pi = None, A = None, B = None):
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
		for obs in reversed(obs_sequence):
			log_beta.append([ logsumexp(np.log(self.A[i,:])+np.log(self.B[:,obs])+log_beta[-1]) for i in xrange(self.n_states)])
		log_beta.reverse()
		return np.array(log_beta)

	def baum_welch(self,obs_sequence):
		NotImplementedError



def generate_observations(model_name, T):
	"""
	The Ride model from west Philly to Engineering.
	State : Chesnut St., Walnut St., Spruce St., Pine St. 
	Observation : Students (five - S, W, P, W, C) 
	model_name : name of a model
	T : length of a observation sequence to generate
	"""
	if model_name == 'uber':
		A = np.array([[0.4, 0.4, 0.1, 0.1],
						[0.3, 0.3, 0.3, 0.1],
						[0.1, 0.3, 0.3, 0.3],
						[0.1, 0.1, 0.4, 0.4]], dtype=np.float32)

		B = np.array([[0.0, 0.2, 0.0, 0.2, 0.6],
						[0.1, 0.4, 0.0, 0.4, 0.1],
						[0.5, 0.2, 0.1, 0.2, 0.0],
						[0.3, 0.1, 0.5, 0.1, 0.0]], dtype=np.float32)

		Pi = np.array([0.3, 0.4, 0.1, 0.2], dtype=np.float32)

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
	return obs_sequence


def inv_sampling(pdf):
	r = np.random.rand() 
	for (i,p) in enumerate(np.cumsum(pdf)):
		if r <= p:
			return i








