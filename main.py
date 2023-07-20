#implmentation of BMF for time series imputation

import numpy as np
from scipy.stats import wishart, multivariate_normal
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
import matplotlib.pyplot as plt
from scipy.stats import expon as exp
import pandas as pd
import openpyxl as xl

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

    
def exponential_prior(U, V, Gamma_U, Gamma_V,dim1, dim2):
    U_prior = np.prod([exp.pdf(U[i, :], scale=1/Gamma_U) for i in range(dim1)])
    V_prior = np.prod([exp.pdf(V[:, j], scale=1/Gamma_V) for j in range(dim2)])
    return U_prior, V_prior


def gaussian_prior(U, V, mu_U, lambda_U, mu_V, lambda_V,dim1, dim2):
    U_prior = np.prod([mvn.pdf(U[i, :], mu_U, lambda_U) for i in range(dim1)])
    V_prior = np.prod([mvn.pdf(V[:, j], mu_V, lambda_V) for j in range(dim2)])
    return U_prior, V_prior

def sample_hyperparameters(U, H0, beta0, d0, mu0):
    N, r = U.shape
    U_bar = np.mean(U, axis=0)
    S_bar = np.dot(U.T, U) / N
    S0_inv = np.linalg.inv(H0["S0"])
    
    beta_star = beta0 + N
    d_star = d0 + N
    mu_star = (beta0 * mu0 + N * U_bar) / beta_star
    S_star_inv = S0_inv + N * S_bar + (beta0 * N / beta_star) * np.outer((U_bar - mu0), (U_bar - mu0))
    S_star = np.linalg.inv(S_star_inv)
    
    lambda_U = wishart.rvs(df=d_star, scale=S_star)
    mu_U = multivariate_normal.rvs(mean=mu_star, cov=np.linalg.inv(beta_star * lambda_U))
    
    return {"mu": mu_U, "lambda": lambda_U}

def sample_latent_feature_vector(X, V, Hi, sigma, mu_U, lambda_U):
    N, r = V.shape
    VV = np.dot(V.T, V)
    XVT = np.dot(X, V.T)
    lambda_star = lambda_U + sigma * VV
    lambda_star_inv = np.linalg.inv(lambda_star)
    mu_star = lambda_star_inv @ (mu_U @ lambda_U + sigma * XVT)
    return multivariate_normal.rvs(mean=mu_star, cov=lambda_star_inv)

# Load the .xlsx file as a pandas DataFrame
df = pd.read_excel('C:/Users/Rohit/Documents/Exeter-Placement/Archive Data/Gen_Demand_Data_Sc3_Chausey_Scenario1.xlsx')

# Convert the DataFrame to a numpy array
X = df.values
# Set the initial values for U0 and V0
N, T = X.shape  # Assume X is your data matrix
r = 10  # Set the rank
U0 = np.random.normal(size=(N, r))
V0 = np.random.normal(size=(r, T))

# Set the prior hyperparameters
H0 = {"mu": np.zeros(r), "lambda": np.eye(r), "S0": np.eye(r)}
beta0 = 1
d0 = r
mu0 = np.zeros(r)
sigma = 1  # Set sigma to 1, adjust this value according to your problem

# Run the Gibbs sampling algorithm
Imax = 10  # Set the number of iterations
U, V = BPMF(U0, V0, X, H0, Imax, beta0, d0, mu0, sigma)

print("yay")
# Now, U and V are the sampled latent feature matrices




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
    
#test_gaussian_prior()
print("yay")
