import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# a function to load tha pictures for the digits


def load_data():
    '''
    First Step: Load and explore the datasets
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