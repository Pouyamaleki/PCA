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
    Inumpyut:
        X: feature matrix (m * n)
        mean_vector: mean of each feature
    Returns:
        X_centered: centerd data matrix
        mean_vector: mean of rach feature
    '''

    # subtract the mean of each sample
    X_centered = X - mean_vector

    return X_centered


def Covariance_matrix(X_centered):
    '''
    Step 3: Compute the covariance matrix
    Inumpyut:
        X_centered: centered data matrix (m * n)
    Returns:
        C: Covariance matrix (m * n)
        is_symmetric: is covariance matrix symmetric or no
    '''

    # get the row of the Centered matrix
    m = X_centered.shape[0]

    # Compute the covariance matrix
    C = (1 / (m-1)) * numpy.dot(X_centered.t, X_centered)

    # check if the covariance matrix is symmetric or no
    is_symmetric = numpy.allclose(C, C.t)

    return C, is_symmetric


def QR_algorithm(C, num_iteration=3):
    '''
    Step 4: QR algorithm for eigenvalue computation
    Inumpyut:
        C: Covariance matrix (m * n)
        num_iteration: number of QR iteration to perform
    Returns:
        C_current: matrix after QR iteration
        Q_matrices: list of Q matrices from each iteration
        R_matrices: list of R matrices from each iteration
        is_similar: a varibale that check the similarity
    '''

    # create and get the needed matrices
    n = C.shape[0]
    C_current = C.copy()
    Q_matrices = []
    R_matrices = []

    # a for loop to decomposition QR {num_iteration} times
    for _ in range(num_iteration):
        # QR decomposition -> C = Q * R (reduced mode)
        Q, R = numpy.linalg.qr(C_current, "reduced")

        # store Q & R for analysis
        Q_matrices.append(Q)
        R_matrices.append(R)

        # update the C matrix
        C_new = numpy.dot(R, Q)

        # check the similarity
        is_similar = numpy.allclose(
            C_new, numpy.dot(numpy.dot(Q.t, Q), numpy.eye(n)))

        return C_current, Q_matrices, R_matrices, is_similar


def demo_on_small_matrix():
    """
    Demo QR algorithm on a small symmetric matrix (4x4)
    """
    print("\n" + "=" * 50)
    print("QR Algorithm Demo on 4x4 Symmetric Matrix")
    print("=" * 50)

    # Create a random symmetric 4x4 matrix
    numpy.random.seed(42)
    A = numpy.random.randn(4, 4)
    A = numpy.dot(A, A.T)  # Make it symmetric positive definite

    print("original matrix A:")
    print(A)
    print(f"\nEigenvalues of A: {numpy.linalg.eigvalsh(A)}")

    # perform one QR iteration
    Q, R = numpy.linalg.qr(A, mode='reduced')
    A1 = numpy.dot(R, Q)

    print("\nafter one QR iteration:")
    print("Q (orthogonal):")
    print(Q)
    print("\nR (upper triangular):")
    print(R)
    print("\nA1 = R * Q:")
    print(A1)
    print(f"Eigenvalues of A1: {numpy.linalg.eigvalsh(A1)}")

    # Check if A1 is similar to A
    is_similar = numpy.allclose(A1, numpy.dot(numpy.dot(Q.T, A), Q))
    print(f"\nIs A1 similar to A? {is_similar}")

    # Rank and nullity
    rank = numpy.linalg.matrix_rank(A)
    nullity = A.shape[0] - rank
    print(f"\nRank of A: {rank}")
    print(f"Nullity of A: {nullity}")

    # Check linear independence of columns of Q
    print(
        f"\nAre columns of Q linearly independent? {numpy.linalg.matrix_rank(Q) == Q.shape[1]}")
    print("=" * 50)


def eigen_decomposition(C):
    '''
    Step 5: Compute eigenvalues and eigenvectors of covariance matrix
    Input:
        C: Covatiance matrix (m * n)
    Return:
        eigenvalues: sorted eigen values in descending order
        eigenvectors: sorted eigenvectors corresponding to eigenvalues
    '''

    # compute eigenvalues and eigenvectors using eigh
    eigenvalues, eigenvectors = numpy.linalg.eigh(C)

    # sort eigen values in descending order
    index = numpy.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]

    return eigenvalues, eigenvectors


def explained_variance(eigenvalues):
    '''
    Step 6: Calculate and visualize explained variance ratio
    Input:
        eigenvalues: sorted eigenvalues in descending order
    Returns:
        explained_variance_ratio: ratio of each eigenvalue to total sum (n,)
        cumulative_variance: cumulative sum of explained variance ratios (n,)
        k_90: number of components needed to preserve 90% variance
    '''

    # Calculate explained variance ratio
    total_variance = numpy.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance

    # Calculate cumulative_variance
    cumulative_variance = numpy.cumsum(explained_variance_ratio)

    # find number of components for 90% variance
    k_90 = numpy.argmax(cumulative_variance >= 0.90) + 1

    return explained_variance_ratio, cumulative_variance, k_90

def dimention_reduction(X_centered, eigenvectors, k):
    '''
    Step 7: Reduce dimentionality by projecting data in to k eigenvector
    Input:
        X_centered: centered data matrix (m * n)
        eigenvectors: sorted eigenvectors matrix (n * n)
        k: number of dimention to keep
    Returns:
        W: projection matrix (n * k)
        T: projected data (m * k)
    '''
    
    # select first k eigen vectors
    W = eigenvectors[:, :, k]
    
    # project data in to new subspace
    T = numpy.dot(X_centered, W)
    
    return W, T
    