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

    return X, Y, mean_vector


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

    # get the row of the Centered matrix
    m = X_centered.shape[0]

    # Compute the covariance matrix
    C = (1 / (m-1)) * numpy.dot(X_centered.T, X_centered)

    # check if the covariance matrix is symmetric or no
    is_symmetric = numpy.allclose(C, C.T)

    return C, is_symmetric


def QR_algorithm(C, num_iteration=3):
    '''
    Step 4: QR algorithm for eigenvalue computation
    Input:
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
            C_new, numpy.dot(numpy.dot(Q.T, Q), numpy.eye(n)))

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


def visualize_2D(X_centered, eigenvectors, y):
    '''
    Step 8: Visualize data in 20 space using first 2 principal componenet
    Input:
        X_centered: centered data matrix (m * n)
        eigenvectors: sorted eigenvectors matrix (n * n)
        y: labels
    '''

    # select first 2 aigenvectors
    W2 = eigenvectors[:, :, 2]

    # project data into 2D space
    T2 = numpy.dot(X_centered, W2)

    # create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(T2[:, 0], T2[:, 1], c=y,
                          cmap="tab10", alpha=0.7, s=30)
    plt.colorbar(scatter, label="Digit Label")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title("PCA Visualization: Digits in 2D Space")
    plt.grid(True, alpha=0.3)
    plt.show()


def reconstruction_error(X, X_centered, eigenvectors, mean_vector, y):
    """
    Step 9: Reconstruct data and analyze reconstruction error
    Input:
        X: original data matrix (m * n)
        X_centered: centered data matrix (m * n)
        eigenvectors: sorted eigenvectors matrix (n * n)
        mean_vector: mean of each feature
        y: labels
    Returns:
        errors: list of MSE errors for different k values
    """

    # Test different k values
    k_values = [1, 5, 10, 15, 20, 30, 40, 50, 64]
    errors = []

    for i in k_values:
        # Select first k eigenvectors
        W = eigenvectors[:, :i]

        # Project data
        T = numpy.dot(X_centered, W)

        # Reconstruct data
        X_reconstructed = numpy.dot(T, W.T) + mean_vector

        # Calculate Mean Squared Error (MSE)
        mse = numpy.mean((X - X_reconstructed) ** 2)
        errors.append(mse)

    # Plot reconstruction error
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, errors, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of Components (k)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Reconstruction Error vs Number of Components")
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    plt.show()

    # Show sample reconstruction for k=10 and k=30
    show_sample_reconstruction(
        X, X_centered, eigenvectors, mean_vector, y, k=10)
    show_sample_reconstruction(
        X, X_centered, eigenvectors, mean_vector, y, k=30)

    return errors


def show_sample_reconstruction(X, X_centered, eigenvectors, mean_vector, y, k, sample_index=0):
    """
    Show original vs reconstructed image for a specific sample
    Input:
        X: original data matrix (m x n)
        X_centered: centered data matrix (m x n)
        eigenvectors: sorted eigenvectors matrix (n x n)
        mean_vector: mean of each feature (n,)
        y: labels (m,)
        k: number of components to use
        sample_index: index of sample to display
    """

    # Select first k eigenvectors
    W = eigenvectors[:, :k]

    # Project and reconstruct
    T = numpy.dot(X_centered[sample_index], W)
    X_reconstructed = numpy.dot(T, W.T) + mean_vector

    # Reshape to 8x8
    original_image = X[sample_index].reshape(8, 8)
    reconstructed_image = X_reconstructed.reshape(8, 8)

    # Plot
    plt.figure(figsize=(8, 4))

    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title(f"Original Image\nLabel: {y[sample_index]}")
    plt.axis("off")

    # Reconstructed
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap="gray")
    plt.title(f"Reconstructed with k={k}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def m_less_than_n(X, y):
    """
    Step 10: Analyze the case where m < n (fewer samples than features)
    Input:
        X: original data matrix (m x n)
        y: labels (m,)
    """
    
    # Randomly select a subset of samples (m < n)
    m_subset = 50  # Choose 50 samples (less than 64 features)
    numpy.random.seed(42)
    indices = numpy.random.choice(X.shape[0], m_subset, replace=False)
    X_subset = X[indices, :]

    # Center the subset data
    mean_vector_subset = numpy.mean(X_subset, axis=0)
    X_centered_subset = X_subset - mean_vector_subset

    # Compute covariance matrix
    m, n = X_centered_subset.shape
    C_subset = (1 / (m - 1)) * numpy.dot(X_centered_subset.T, X_centered_subset)

    # Compute eigenvalues
    eigenvalues_subset, eigenvectors_subset = numpy.linalg.eigh(C_subset)

    # Sort eigenvalues in descending order
    idx = numpy.argsort(eigenvalues_subset)[::-1]
    eigenvalues_subset = eigenvalues_subset[idx]
    eigenvectors_subset = eigenvectors_subset[:, idx]

    # Count zero (or near-zero) eigenvalues
    tolerance = 1e-10

    # Visualize eigenvalues
    plt.figure(figsize=(12, 5))

    # Plot 1: All eigenvalues
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(eigenvalues_subset) + 1), eigenvalues_subset)
    plt.axhline(y=tolerance, color='r', linestyle='--',
                label=f'Tolerance = {tolerance}')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title(f'Eigenvalues (m={m_subset}, n={n})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Non-zero eigenvalues only
    non_zero_eigenvalues = eigenvalues_subset[eigenvalues_subset >= tolerance]
    plt.subplot(1, 2, 2)
    plt.bar(range(1, len(non_zero_eigenvalues) + 1), non_zero_eigenvalues)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title(f'Non-zero Eigenvalues ({len(non_zero_eigenvalues)} components)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return eigenvalues_subset, eigenvectors_subset, X_centered_subset


def main():
    """
    Main function: Execute all steps of the PCA project
    """
    print("=" * 60)
    print("PCA Project - Linear Algebra Final Project")
    print("=" * 60)
    
    # ===== Step 1: Load Data =====
    X, y, mean_vector = load_data()
    print(f"Step 1: Loaded data with shape {X.shape}")
    
    # ===== Step 2: Center Data =====
    X_centered = Center_data(X, mean_vector)
    print(f"Step 2: Centered data with shape {X_centered.shape}")
    
    # ===== Step 3: Covariance Matrix =====
    C = Covariance_matrix(X_centered)
    print(f"Step 3: Covariance matrix shape {C.shape}")
    
    # ===== Step 4: QR Algorithm =====
    C_final, Q_list, R_list = QR_algorithm(C, num_iterations=3)
    print(f"Step 4: QR algorithm completed with {len(Q_list)} iterations")
    
    # QR Demo on small matrix
    A, Q, R, A1 = demo_on_small_matrix()
    print(f"Step 4 Demo: QR on 4x4 matrix completed")
    
    # ===== Step 5: Eigen Decomposition =====
    eigenvalues, eigenvectors = eigen_decomposition(C)
    print(f"Step 5: Eigen decomposition completed - {len(eigenvalues)} eigenvalues")
    
    # ===== Step 6: Explained Variance =====
    explained_variance_ratio, cumulative_variance, k_90 = explained_variance(eigenvalues)
    print(f"Step 6: 90% variance preserved with {k_90} components")
    explained_variance(explained_variance_ratio, cumulative_variance, k_90)
    
    # ===== Step 7: Dimensionality Reduction =====
    k = 10
    W, T = dimention_reduction(X_centered, eigenvectors, k)
    print(f"Step 7: Reduced dimensions from {X_centered.shape[1]} to {k}")
    
    # ===== Step 8: 2D Visualization =====
    T2 = visualize_2D(X_centered, eigenvectors, y)
    print(f"Step 8: 2D visualization completed")
    
    # ===== Step 9: Reconstruction Error =====
    errors = reconstruction_error(X, X_centered, eigenvectors, mean_vector, y)
    print(f"Step 9: Reconstruction error analysis completed")
    
    # ===== Step 10: m < n Case =====
    eigenvalues_subset, eigenvectors_subset, X_centered_subset = m_less_than_n(X, y)
    print(f"Step 10: m < n case analysis completed")
    
    print("=" * 60)
    print("All steps completed successfully!")
    print("=" * 60)
    
    # Return all important variables for further analysis if needed
    return {
        'X': X,
        'y': y,
        "X_centered": X_centered,
        "mean_vector": mean_vector,
        'C': C,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
        "k_90": k_90,
        "W": W,
        "T": T,
        "T2": T2,
        "errors": errors
    }