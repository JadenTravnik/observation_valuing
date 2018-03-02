import numpy as np
import math
import random



class Verifier:
	def __init__(self, gammaTime):
		self.z = []
		self.gamma = []
		self.gammaTime = gammaTime
		self.Return = []
		self.predictions = [0]*gammaTime

	def update(self, _z, _gamma, prediction):
		self.z.append(_z)
		self.gamma.append(_gamma)

		self.predictions.append(prediction)

		if len(self.z) == self.gammaTime:
			self.Return.append(sum([np.product(self.gamma[:t])*self.z[t] for t in range(self.gammaTime)]))
			self.z.pop(0)
			self.gamma.pop(0)
		else:
			self.Return.append(0)

		pred = self.predictions.pop(0)

		return self.Return[-1], pred

class VerifierGroup:
	def __init__(self):
		self.verifiers = []

	def addVerifier(v):
		self.verifiers.append(v)

	def update(self, Z, Gamma):
		for i in range(len(self.verifiers)):
			self.verifiers[i].update(Z[i], Gamma[i])

	def getReturn(self):
		return [v.Return for v in self.verifiers]




class OnPolicyGVF:

	def __init__(self, gamma_id=0, phi_id = 0, lamda=0.99, alpha=0.1, numFeatures=10, numActiveFeatures=1, id=0, doRupee=False, doUde=False):
		self.gamma_id = gamma_id
		self.alpha = alpha
		self.lamda = lamda
		self.id = id
		self.phi_id = phi_id
		self.prediction_t = 0

		self.numFeatures = numFeatures

		if doRupee:

			def resetRupee(self):
				self.h = np.array(np.zeros(self.numFeatures))
				self.alpha_h = 5*alpha
				self.delta_e = np.array(np.zeros(self.numFeatures))
				self.rupee_beta = 0
				self.tau = 0
				self.rupee_init_beta = (1-lamda)*self.alpha_h/self.numActiveFeatures
				self.rupee = 0

			setattr(self.__class__, 'resetRupee', resetRupee)


			def updateRupee(self):
				self.h = self.h  + self.alpha_h*(self.delta*self.e - np.dot(self.h,self.phi)*self.phi)
				self.tau = (1-self.rupee_init_beta)*self.tau + self.rupee_init_beta
				self.rupee_beta = self.rupee_init_beta/float(self.tau)
				self.delta_e = (1-self.rupee_init_beta)*self.delta_e + self.rupee_beta*self.delta*self.e
				self.rupee = math.sqrt(math.fabs(np.dot(self.h,self.delta_e)))

			setattr(self.__class__, 'updateRupee', updateRupee)

		else:

			def resetRupee(self):
				self.rupee = 0
				return
			setattr(self.__class__, 'resetRupee', resetRupee)

			def updateRupee(self):
				return
			setattr(self.__class__, 'updateRupee', updateRupee)


		if doUde:

			def resetUde(self):
				self.delta_mean = 0.0
				self.mean_diff_trace = 0.0
				self.ude = 0.0
				self.delta_trace = 0.0
				self.count = 1
				self.ude_beta = alpha*10
				self.inv_ude_beta = 1-self.ude_beta
				self.ude_epsilon = 0.00001

			setattr(self.__class__, 'resetUde', resetUde)

			def updateUde(self):
				self.delta_trace = self.delta_trace*self.inv_ude_beta + self.delta*self.ude_beta

				diff = self.delta - self.delta_mean
				self.delta_mean += diff/float(self.count)
				self.mean_diff_trace += diff*(self.delta - self.delta_mean)
				try:
					self.ude = math.fabs(self.delta_trace / (math.sqrt(self.mean_diff_trace / (self.count - 1.0))+self.ude_epsilon))
				except:
					pass

			setattr(self.__class__, 'updateUde', updateUde)

		else:
			def resetUde(self):
				self.ude = 0.0
				return
			setattr(self.__class__, 'resetUde', resetUde)

			def updateUde(self):
				return
			setattr(self.__class__, 'updateUde', updateUde)

		self.reset()

	def reset(self):
		self.phi = np.array(np.zeros(self.numFeatures))
		self.phi_next = np.array(np.zeros(self.numFeatures))
		self.w = np.array(np.zeros(self.numFeatures))
		self.e = np.array(np.zeros(self.numFeatures))

		self.resetRupee()
		self.resetUde()

	def update(self, phi, phi_next, z, gamma_t, gamma_t_1):

		self.phi = phi
		self.phi_next = phi_next

		self.z = z
		prediction_t_1 = np.dot(phi_next,self.w)

		self.delta = z + gamma_t_1*prediction_t_1 - self.prediction_t
		self.e = gamma_t*self.lamda*self.e + phi*self.alpha - self.alpha*gamma_t*self.lamda*np.dot(self.e,phi)*phi

		self.w += self.delta*self.e + self.alpha*(self.prediction_t - np.dot(phi, self.w))*phi

		self.updateRupee()
		self.updateUde()


		self.prediction_t = prediction_t_1
		self.normalized_prediction = self.prediction_t*(1-gamma_t_1)
		self.prediction = self.prediction_t
		return self.prediction_t

	def predict(self, phi):
		return np.dot(phi, self.w)

class OffPolicyGVF:

	def __init__(self, t_policy_id, z_id, gamma_id, phi_id=0, lamda=0.99, alpha=.01, beta=.001, numFeatures=10, numActiveFeatures=1, doRupee=False, doUde=False):

		self.beta = beta
		self.lamda = lamda
		self.inv_lamda = 1. - lamda
		self.alpha = alpha

		self.t_policy_id = t_policy_id
		self.z_id = z_id
		self.gamma_id = gamma_id
		self.phi_id = phi_id

		self.numFeatures = numFeatures
		self.prediction = 0
		self.normalized_prediction = 0
		self.prevRho = 0

		if doRupee:

			def resetRupee(self):
				self.h = np.array(np.zeros(self.numFeatures))
				self.alpha_h = 5*alpha
				self.delta_e = np.array(np.zeros(self.numFeatures))
				self.rupee_beta = 0
				self.tau = 0
				self.rupee_init_beta = 0.001
				self.rupee = 0

			setattr(self.__class__, 'resetRupee', resetRupee)


			def updateRupee(self):
				self.h = self.h  + self.alpha_h*(self.delta*self.e - np.dot(self.h,self.phi)*self.phi)
				self.tau = (1-self.rupee_init_beta)*self.tau + self.rupee_init_beta
				self.rupee_beta = self.rupee_init_beta/float(self.tau)
				self.delta_e = (1-self.rupee_init_beta)*self.delta_e + self.rupee_beta*self.delta*self.e
				self.rupee = math.sqrt(math.fabs(np.dot(self.h,self.delta_e)))

			setattr(self.__class__, 'updateRupee', updateRupee)

		else:

			def resetRupee(self):
				self.rupee = 0
				return
			setattr(self.__class__, 'resetRupee', resetRupee)

			def updateRupee(self):
				return
			setattr(self.__class__, 'updateRupee', updateRupee)


		if doUde:

			def resetUde(self):
				self.delta_mean = 0.0
				self.mean_diff_trace = 0.0
				self.ude = 0.0
				self.delta_trace = 0.0
				self.count = 1
				self.ude_beta = alpha*10
				self.inv_ude_beta = 1-self.ude_beta
				self.ude_epsilon = 0.00001

			setattr(self.__class__, 'resetUde', resetUde)

			def updateUde(self):
				self.delta_trace = self.delta_trace*self.inv_ude_beta + self.delta*self.ude_beta

				diff = self.delta - self.delta_mean
				self.delta_mean += diff/float(self.count)
				self.mean_diff_trace += diff*(self.delta - self.delta_mean)
				try:
					self.ude = math.fabs(self.delta_trace / (math.sqrt(self.mean_diff_trace / (self.count - 1.0))))
				except:
					self.ude = 0
					pass
				self.count += 1

			setattr(self.__class__, 'updateUde', updateUde)

		else:
			def resetUde(self):
				self.ude = 0.0
				return
			setattr(self.__class__, 'resetUde', resetUde)

			def updateUde(self):
				return
			setattr(self.__class__, 'updateUde', updateUde)

		self.reset()

	def reset(self):
		self.phi = np.array(np.zeros(self.numFeatures))
		self.phi_next = np.array(np.zeros(self.numFeatures))
		self.w = np.array(np.zeros(self.numFeatures))
		self.e = np.array(np.zeros(self.numFeatures))
		self.e_gradient = np.array(np.zeros(self.numFeatures))
		self.e_w = np.array(np.zeros(self.numFeatures))
		self.th = np.array(np.zeros(self.numFeatures))
		self.prevTh = np.array(np.zeros(self.numFeatures))
		self.resetRupee()
		self.resetUde()

	def update(self, phi, phi_next, z, rho, gamma_t, gamma_t_1):
		self.phi_next = phi_next
		self.z = z
		self.phi = phi


		self.prediction = np.dot(self.phi,self.th)
		self.normalized_prediction = self.prediction*(1-gamma_t_1)


		self.delta = z + gamma_t_1*np.dot(self.phi_next,self.th) - self.prediction

		# print('delta = z + gamma_t_1*np.dot(self.phi_next,self.th)  -  np.dot(self.phi,self.th)')
		# print(str(self.delta) + ' = ' + str(z) + ' + ' + str(gamma_t_1) + ' * ' + str(np.dot(self.phi_next,self.th)) + ' - ' + str(self.prediction) + '\n' \
		# 	+ '  |phi|:' + str(sum(phi)) + '  |phi_next|:' + str(sum(phi_next)) \
		# 	+ '  z: ' + str(z) + '  rho: ' + str(rho) \
		# 	+ '  g_t: ' + str(gamma_t) + '  g_t_1: ' + str(gamma_t_1) + '  delta: ' + str(self.delta))

		self.e = rho*(gamma_t*self.lamda*self.e + self.alpha*(1- rho*gamma_t*self.lamda*phi.T*self.e)*phi)
		self.e_gradient = rho*(gamma_t*self.lamda*self.e_gradient + phi)

		self.e_w = self.prevRho*gamma_t*self.lamda*self.e_w + self.beta*(1- self.prevRho*gamma_t*self.lamda*phi.T*self.e_w)*self.phi

		tempTh = self.th
		self.th += self.delta*self.e \
				+ (self.e - self.alpha*rho*phi)*((self.th - self.prevTh).T*phi) \
				- self.alpha*gamma_t_1*(1-self.lamda)*self.w.T*self.e_gradient*phi_next


		self.w += rho*self.delta*self.e_w - self.beta*phi.T*self.w*phi

			#self.alpha*(self.delta*self.e - gamma_t_1*self.inv_lamda*np.dot(self.e, self.w)*self.phi_next)


		#self.w += rho*self.delta*self.e - self.beta*np.dot(self.phi,self.w)*self.phi


		self.prevRho = rho
		self.prevTh = tempTh

		self.updateRupee()
		self.updateUde()
		return self.prediction

	def predict(self, phi):
		return np.dot(self.phi, self.th)


class Horde:
	def __init__(self):
		self.demons = []

	def resetTraces(self):
		for d in self.demons:
			d.e *= 0

	def addDemon(self, d):
		d.id = len(self.demons)
		self.demons.append(d)

	def getPreds(self):
		return np.array([d.normalized_prediction for d in self.demons])

	def update(self, Phi, Phi_next, Z, Gamma,  Gamma_t_1):
		preds = []
		for d in self.demons:
			p = d.update(Phi[d.phi_id], Phi_next[d.phi_id], Z[d.id], Gamma[d.gamma_id], Gamma_t_1[d.gamma_id])
			preds.append(p)
		return preds
	def getRupee(self):
		return np.array([d.rupee for d in self.demons])

	def getUde(self):
		return np.array([d.ude for d in self.demons])

class OffPolicyHorde(Horde):

	def update(self, Phi, Phi_next, Z, Rho, Gamma_t, Gamma_t_1):
		for d in self.demons:
			d.update(Phi[d.phi_id], Phi_next[d.phi_id], Z[d.z_id], Rho[d.t_policy_id], Gamma_t[d.gamma_id], Gamma_t_1[d.gamma_id])
