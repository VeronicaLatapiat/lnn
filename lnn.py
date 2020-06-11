#Written by Weihao Gao from UIUC

import scipy.spatial as ss
import scipy.stats as sst
import scipy.io as sio
from scipy.special import beta,digamma,gamma
from sklearn.neighbors import KernelDensity
from math import log,pi,exp
import numpy.random as nr
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from cvxopt import matrix,solvers


#Usage Functions

#This is the main entropy estimator
def entropy(x,k=5,tr=30,bw=0):
	return LNN_2_entropy(x,k,tr,bw)

def LNN_2_entropy(x,k=5,tr=30,bw=0):

	'''
		Estimate the entropy H(X) from samples {x_i}_{i=1}^N
		Using Local Nearest Neighbor (LNN) estimator with order 2

		Input: x: 2D list of size N*d_x
		k: k-nearest neighbor parameter
		tr: number of sample used for computation
		bw: option for bandwidth choice, 0 = kNN bandwidth, otherwise you can specify the bandwidth

		Output: one number of H(X)
	'''

	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	assert tr <= len(x)-1, "Set tr smaller than num.samples - 1"
	N = len(x)
   	d = len(x[0])
	
	local_est = np.zeros(N)
	S_0 = np.zeros(N)
	S_1 = np.zeros(N)
	S_2 = np.zeros(N)
	tree = ss.cKDTree(x)
	if (bw == 0):
		bw = np.zeros(N)
	for i in range(N):
		lists = tree.query(x[i],tr+1,p=2)
		knn_dis = lists[0][k]
		list_knn = lists[1][1:tr+1]
		
		if (bw[i] == 0):
			bw[i] = knn_dis

		S0 = 0
		S1 = np.matrix(np.zeros(d))
		S2 = np.matrix(np.zeros((d,d)))
		for neighbor in list_knn:
			dis = np.matrix(x[neighbor] - x[i])
			S0 += exp(-dis*dis.transpose()/(2*bw[i]**2))
			S1 += (dis/bw[i])*exp(-dis*dis.transpose()/(2*bw[i]**2))
			S2 += (dis.transpose()*dis/(bw[i]**2))*exp(-dis*dis.transpose()/(2*bw[i]**2))

		Sigma = S2/S0 - S1.transpose()*S1/(S0**2)
		det_Sigma = np.linalg.det(Sigma)
		if (det_Sigma < (1e-4)**d):
			local_est[i] = 0
		else:
			offset = (S1/S0)*np.linalg.inv(Sigma)*(S1/S0).transpose()
			local_est[i] = -log(S0) + log(N-1) + 0.5*d*log(2*pi) + d*log(bw[i]) + 0.5*log(det_Sigma) + 0.5*offset[0][0]

	if (np.count_nonzero(local_est) == 0):
		return 0
	else: 
		return np.mean(local_est[np.nonzero(local_est)])


#These is main mutual information estimator
def mi(data,split,k=5,tr=30):
	return _3LNN_2_mi(data,split,k,tr)

def _3LNN_2_mi(data,split,k=5,tr=30):
	'''
		Estimate the mutual information I(X;Y) from samples {x_i,y_i}_{i=1}^N
		Using I(X;Y) = H_{LNN}(X) + H_{LNN}(Y) - H_{LNN}(X;Y)
		where H_{LNN} is the LNN entropy estimator with order 2

		Input: data: 2D list of size N*(d_x + d_y)
		split: should be d_x, splitting the data into two parts, X and Y
		k: k-nearest neighbor parameter
		tr: number of sample used for computation

		Output: one number of I(X;Y)
	'''
	assert split >=1, "x must have at least one dimension"
	assert split <= len(data[0]) - 1, "y must have at least one dimension"
	x = data[:,:split]
	y = data[:,split:]

	N = len(data)
	return LNN_2_entropy(x,k,tr) + LNN_2_entropy(y,k,tr) - LNN_2_entropy(data,k,tr)

#Auxilary Functions

def vd(d,q):
	# Return the volume of unit q-norm ball in d dimension space
	if (q==float('inf')):
		return d*log(2)
	return d*log(2*gamma(1+1.0/q)) - log(gamma(1+d*1.0/q))

## Entropia de orden 1
def LNN_1_entropy(x,k=5,tr=30,bw = 0):
	'''
		Estimate the entropy H(X) from samples {x_i}_{i=1}^N
		Using Local Nearest Neighbor (LNN) estimator with order 1

		Input: x: 2D list of size N*d_x
		k: k-nearest neighbor parameter
		tr: number of sample used for computation
		bw: option for bandwidth choice, 0 = kNN bandwidth, otherwise you can specify the bandwidth

		Output: one number of H(X)
	'''
	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	assert tr <= len(x)-1, "Set tr smaller than num.samples - 1"
	N = len(x)
   	d = len(x[0])

	local_est = np.zeros(N)
	S_0 = np.zeros(N)
	S_1 = np.zeros(N)
	S_2 = np.zeros(N)
	tree = ss.cKDTree(x)
	if (bw == 0):
		bw = np.zeros(N)
	for i in range(N):
		lists = tree.query(x[i],tr+1,p=2)
		knn_dis = lists[0][k]
		list_knn = lists[1][1:tr+1]
		
		if (bw[i] == 0):
			bw[i] = knn_dis
			#bw = 1.06*(N**(-1.0/(d+4)))
		S0 = 0
		S1 = np.matrix(np.zeros(d))
		for neighbor in list_knn:
			dis = np.matrix(x[neighbor] - x[i])
			S0 += exp(-dis*dis.transpose()/(2*bw[i]**2))
			S1 += (dis/bw[i])*exp(-dis*dis.transpose()/(2*bw[i]**2))
		
		offset = (S1/S0)*(S1/S0).transpose()
		local_est[i] = -log(S0) + log(N) + 0.5*d*log(2*pi) + d*log(bw[i]) + 0.5*offset[0][0]
		if (abs(local_est[i]) > (1e+4)**d):
			local_est[i] = 0

	if (np.count_nonzero(local_est) == 0):
		return 0
	else: 
		return np.mean(local_est[np.nonzero(local_est)])

