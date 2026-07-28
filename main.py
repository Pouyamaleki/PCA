import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def load_data():
    '''
    Step 1: Load and explore the datasets
    Returns:
        X: feature matrix
        Y: labels
        m: number of samples
        n: number of features
        first_sample: first picture as a vector
        mean_vector: mean of each feature
    '''

    # Load the digits datasets
    digits = load_digits()
    X = digits.data   # Feature matrix
    Y = digits.target  # Labels

    # get the dimentions
    m, n = X.shape

    # get the first image
    first_sample = X[0]

    # compute mean of each feature
    mean_vector = numpy.mean(X, 0)

    return X, Y, m, n, first_sample, mean_vector


def Center_data(X, mean_vector):
    '''
    Step 2: Center the data by subtracting the mean of each feature
    Input:
        X: feature matrix (m * n)
        mean_vector: mean of each feature
    Returns:
        X_centered: centerd data matrix
        mean_vector: mean of rach feature
    '''

    # subtract the mean of each sample
    X_centered = X - mean_vector

    # verify the centring proccess
    column_mean = numpy.mean(X_centered, 0)

    return X_centered


def Covariance_matrix(X_centered):
    '''
    Step 3: Compute the covariance matrix
    Input:
        X_centered: centered data matrix (m * n)
    Returns:
        C: Covariance matrix (m * n)
        is_symmetric: is covariance matrix symmetric or no
    '''

    #
    m, n = X_centered.shape

    # Compute the covariance matrix
    C = (1 / (m-1)) * numpy.dot(X_centered.t, X_centered)

    # check if the covariance matrix is symmetric or no
    is_symmetric = numpy.allclose(C, C.t)

    return C, is_symmetric
