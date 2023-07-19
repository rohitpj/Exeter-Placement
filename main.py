#implmentation of BMF for time series imputation

import numpy as np
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart
from scipy.stats import invwishart
from scipy.stats import multivariate_normal as mvn
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
#import matplotlib.pyplot as plt
from scipy.stats import expon as exp

#stole all these functions hehehe

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

def compute_rmse(var, var_hat):
    return  np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def mat2ten(mat, dim, mode):
    index = list()
    index.append(mode)
    for i in range(dim.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(dim[index]), order = 'F'), 0, mode)

def mvnrnd_pre(mu, Lambda):
    src = normrnd(size = (mu.shape[0],))
    return solve_ut(cholesky_upper(Lambda, overwrite_a = True, check_finite = False), 
                    src, lower = False, check_finite = False, overwrite_b = True) + mu

def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat

def gibbs_sampling(U, V, Imax):
    # Placeholder for the update equations
    def update(U, V):
        # Implement the update equations from the paper here
        return U, V

    for i in range(Imax):
        U, V = update(U, V)
    return np.dot(U, V)

def update_hyper(U, V, Imax):
    print("update_hyper")

def BPMF(dense_mat, sparse_mat, init, rank, burn_iter, gibbs_iter):
    #rohits
    max_iter = burn_iter + gibbs_iter
    dim1, dim2 = sparse_mat.shape
    U = np.random.rand(dim1, rank)
    V = np.random.rand(rank, dim2)
    E = np.zeros((dim1, dim2))

    mu_U = np.zeros(r)  # Mean for the Gaussian prior over U
    lambda_U = np.eye(r)  # Precision matrix for the Gaussian prior over U

    mu_V = np.zeros(r)  # Mean for the Gaussian prior over V
    ambda_V = np.eye(r)  # Precision matrix for the Gaussian prior over V

    def exponential_prior(U, V, Gamma_U, Gamma_V):
        U_prior = np.prod([exp.pdf(U[i, :], scale=1/Gamma_U) for i in range(dim1)])
        V_prior = np.prod([exp.pdf(V[:, j], scale=1/Gamma_V) for j in range(dim2)])
        return U_prior, V_prior
    

    def gaussian_prior(U, V, mu_U, lambda_U, mu_V, lambda_V):
        U_prior = np.prod([mvn.pdf(U[i, :], mu_U, lambda_U) for i in range(dim1)])
        V_prior = np.prod([mvn.pdf(V[:, j], mu_V, lambda_V) for j in range(dim2)])
        return U_prior, V_prior
    
def exponential_prior(U, V, Gamma_U, Gamma_V,dim1, dim2):
    U_prior = np.prod([exp.pdf(U[i, :], scale=1/Gamma_U) for i in range(dim1)])
    V_prior = np.prod([exp.pdf(V[:, j], scale=1/Gamma_V) for j in range(dim2)])
    return U_prior, V_prior


def gaussian_prior(U, V, mu_U, lambda_U, mu_V, lambda_V,dim1, dim2):
    U_prior = np.prod([mvn.pdf(U[i, :], mu_U, lambda_U) for i in range(dim1)])
    V_prior = np.prod([mvn.pdf(V[:, j], mu_V, lambda_V) for j in range(dim2)])
    return U_prior, V_prior

def test_gaussian_prior():
    dim1, dim2, rank = 10, 20, 5  # just some test values
    U = np.random.rand(dim1, rank)
    V = np.random.rand(rank, dim2)

    mu_U = np.zeros(rank)  # Mean for the Gaussian prior over U
    lambda_U = np.eye(rank)  # Precision matrix for the Gaussian prior over U

    mu_V = np.zeros(rank)  # Mean for the Gaussian prior over V
    lambda_V = np.eye(rank)  # Precision matrix for the Gaussian prior over V
    
    U_prior, V_prior = gaussian_prior(U, V, mu_U, lambda_U, mu_V, lambda_V,dim1, dim2)

    # Test if U_prior and V_prior are in valid probability range (0-1)
    assert 0 <= U_prior <= 1, "U_prior is out of valid probability range (0-1)"
    assert 0 <= V_prior <= 1, "V_prior is out of valid probability range (0-1)"
    
test_gaussian_prior()
print("yay")
