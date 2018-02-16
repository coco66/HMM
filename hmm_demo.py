import numpy as np

class HMM():
	def __init__(self,n_states, n_obs):
		self.n_states = n_states # N
		self.n_obs = n_obs # M
		self.Pi = np.random.rand((n_states,), dtype=np.float32)
		self.Pi = self.Pi / np.sum(self.Pi)
		self.A = np.random.rand((n_states, n_states), dtype = np.float32) # transition probability
		self.A = np.divide(self.A, np.sum(self.A, axis=1, keepdims=True)) 
		self.B = np.random.rand((n_states, n_obs), dtype=np.float32) # observation probability (state -> row, obs -> col)
		self.B = np.divide(self.B, np.sum(self.B, axis=1, keepdims=True))

	def forward(self, obs_sequence):
		NotImplementedError
		# return forward_probabilities


	def backward(self,obs_sequence):
		NotImplementedError
		# return backward_probabilities

	def baum_welch(self,obs_sequence):
		NotImplementedError



def generate_observations(model):
	"""
	The Ride model from west Philly to Engineering.
	State : Chesnut St., Walnut St., Spruce St., Pine St. 
	Observation : Students (five - S, W, P, W, C) 
	model : model_name
	"""
	if 'uber':
		A = np.array([[0.4, 0.4, 0.1, 0.1],
						[0.3, 0.3, 0.3, 0.1],
						[0.1, 0.3, 0.3, 0.3],
						[0.1, 0.1, 0.4, 0.4]], dtype=np.float32)

		B = np.array([[0.0, 0.2, 0.0, 0.2, 0.6],
						[0.2, ]], dtype=np.float32)
		Pi = np.array([], dtype=np.float32)
