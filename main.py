import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def load_data():
    '''
    Step 1: Load and explore the datasets
    Returns:
        X: feature matrix
        Y: labels
        mean_vector: mean of each feature
    '''
    digits = load_digits()
    X = digits.data
    Y = digits.target
    mean_vector = numpy.mean(X, 0)
    return X, Y, mean_vector


def Center_data(X, mean_vector):
    '''
    Step 2: Center the data by subtracting the mean of each feature
    '''
    X_centered = X - mean_vector
    return X_centered


def Covariance_matrix(X_centered):
    '''
    Step 3: Compute the covariance matrix
    '''
    m = X_centered.shape[0]
    C = (1 / (m - 1)) * numpy.dot(X_centered.T, X_centered)
    return C


def QR_algorithm(C, num_iterations=3):
    '''
    Step 4: QR algorithm for eigenvalue computation
    Returns:
        C_current: matrix after QR iterations
        Q_matrices: list of Q matrices from each iteration
        R_matrices: list of R matrices from each iteration
    '''
    C_current = C.copy()
    Q_matrices = []
    R_matrices = []

    for _ in range(num_iterations):
        Q, R = numpy.linalg.qr(C_current, "reduced")
        Q_matrices.append(Q)
        R_matrices.append(R)
        C_new = numpy.dot(R, Q)
        C_current = C_new

    return C_current, Q_matrices, R_matrices


def demo_on_small_matrix():
    """
    Demo QR algorithm on a small symmetric matrix (4x4)
    """
    print("\n" + "=" * 50)
    print("QR Algorithm Demo on 4x4 Symmetric Matrix")
    print("=" * 50)

    numpy.random.seed(42)
    A = numpy.random.randn(4, 4)
    A = numpy.dot(A, A.T)  # symmetric positive definite

    print("Original matrix A:\n", A)
    print(f"\nEigenvalues of A: {numpy.linalg.eigvalsh(A)}")

    Q, R = numpy.linalg.qr(A, mode='reduced')
    A1 = numpy.dot(R, Q)

    print("\nAfter one QR iteration:")
    print("Q (orthogonal):\n", Q)
    print("\nR (upper triangular):\n", R)
    print("\nA1 = R * Q:\n", A1)
    print(f"Eigenvalues of A1: {numpy.linalg.eigvalsh(A1)}")

    is_similar = numpy.allclose(A1, numpy.dot(numpy.dot(Q.T, A), Q))
    print(f"\nIs A1 similar to A? {is_similar}")

    rank = numpy.linalg.matrix_rank(A)
    nullity = A.shape[0] - rank
    print(f"\nRank of A: {rank}")
    print(f"Nullity of A: {nullity}")
    print(f"Are columns of Q linearly independent? {numpy.linalg.matrix_rank(Q) == Q.shape[1]}")
    print("=" * 50)

    return A, Q, R, A1


def eigen_decomposition(C):
    '''
    Step 5: Compute eigenvalues and eigenvectors of covariance matrix
    '''
    eigenvalues, eigenvectors = numpy.linalg.eigh(C)
    index = numpy.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]
    return eigenvalues, eigenvectors


def explained_variance(eigenvalues):
    '''
    Step 6: Calculate and visualize explained variance ratio
    Returns:
        explained_variance_ratio: (n,)
        cumulative_variance: (n,)
        k_90: number of components for 90% variance
    '''
    total_variance = numpy.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = numpy.cumsum(explained_variance_ratio)
    k_90 = numpy.argmax(cumulative_variance >= 0.90) + 1

    # Visualization
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Individual Explained Variance')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    plt.axhline(y=0.90, color='k', linestyle='--', label='90% threshold')
    plt.axvline(x=k_90, color='g', linestyle='--', label=f'k={k_90}')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return explained_variance_ratio, cumulative_variance, k_90


def dimension_reduction(X_centered, eigenvectors, k):
    '''
    Step 7: Reduce dimensionality by projecting data onto k eigenvectors
    '''
    W = eigenvectors[:, :k]
    T = numpy.dot(X_centered, W)
    return W, T


def visualize_2D(X_centered, eigenvectors, y):
    '''
    Step 8: Visualize data in 2D space using first 2 principal components
    Returns:
        T2: projected data in 2D
    '''
    W2 = eigenvectors[:, :2]
    T2 = numpy.dot(X_centered, W2)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(T2[:, 0], T2[:, 1], c=y,
                          cmap="tab10", alpha=0.7, s=30)
    plt.colorbar(scatter, label="Digit Label")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title("PCA Visualization: Digits in 2D Space")
    plt.grid(True, alpha=0.3)
    plt.show()

    return T2


def reconstruction_error(X, X_centered, eigenvectors, mean_vector, y):
    """
    Step 9: Reconstruct data and analyze reconstruction error
    """
    k_values = [1, 5, 10, 15, 20, 30, 40, 50, 64]
    errors = []

    for i in k_values:
        W = eigenvectors[:, :i]
        T = numpy.dot(X_centered, W)
        X_reconstructed = numpy.dot(T, W.T) + mean_vector
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

    # Show sample reconstructions for k=10 and k=30
    show_sample_reconstruction(X, X_centered, eigenvectors, mean_vector, y, k=10)
    show_sample_reconstruction(X, X_centered, eigenvectors, mean_vector, y, k=30)

    return errors


def show_sample_reconstruction(X, X_centered, eigenvectors, mean_vector, y, k, sample_index=0):
    """
    Show original vs reconstructed image for a specific sample
    """
    W = eigenvectors[:, :k]
    T = numpy.dot(X_centered[sample_index], W)
    X_reconstructed = numpy.dot(T, W.T) + mean_vector

    original_image = X[sample_index].reshape(8, 8)
    reconstructed_image = X_reconstructed.reshape(8, 8)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title(f"Original Image\nLabel: {y[sample_index]}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap="gray")
    plt.title(f"Reconstructed with k={k}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def m_less_than_n(X, y):
    """
    Step 10: Analyze the case where m < n (fewer samples than features)
    """
    m_subset = 50
    numpy.random.seed(42)
    indices = numpy.random.choice(X.shape[0], m_subset, replace=False)
    X_subset = X[indices, :]

    mean_vector_subset = numpy.mean(X_subset, axis=0)
    X_centered_subset = X_subset - mean_vector_subset

    m, n = X_centered_subset.shape
    C_subset = (1 / (m - 1)) * numpy.dot(X_centered_subset.T, X_centered_subset)

    eigenvalues_subset, eigenvectors_subset = numpy.linalg.eigh(C_subset)
    index = numpy.argsort(eigenvalues_subset)[::-1]
    eigenvalues_subset = eigenvalues_subset[index]
    eigenvectors_subset = eigenvectors_subset[:, index]

    tolerance = 1e-10
    non_zero_count = numpy.sum(eigenvalues_subset >= tolerance)

    # Visualize eigenvalues
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(eigenvalues_subset) + 1), eigenvalues_subset)
    plt.axhline(y=tolerance, color='r', linestyle='--', label=f'Tolerance = {tolerance}')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title(f'Eigenvalues (m={m_subset}, n={n})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    non_zero_eigenvalues = eigenvalues_subset[eigenvalues_subset >= tolerance]
    plt.bar(range(1, len(non_zero_eigenvalues) + 1), non_zero_eigenvalues)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title(f'Non-zero Eigenvalues ({len(non_zero_eigenvalues)} components)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Number of non-zero eigenvalues: {non_zero_count} (out of {n})")
    print(f"Rank of covariance matrix: {non_zero_count}")

    return eigenvalues_subset, eigenvectors_subset, X_centered_subset


def main():
    """
    Main function: Execute all steps of the PCA project
    """
    print("=" * 60)
    print("PCA Project - Linear Algebra Final Project")
    print("=" * 60)

    # Step 1: Load Data
    X, y, mean_vector = load_data()
    print(f"Step 1: Loaded data with shape {X.shape}")

    # Step 2: Center Data
    X_centered = Center_data(X, mean_vector)
    print(f"Step 2: Centered data with shape {X_centered.shape}")

    # Step 3: Covariance Matrix
    C = Covariance_matrix(X_centered)
    print(f"Step 3: Covariance matrix shape {C.shape}")

    # Step 4: QR Algorithm
    C_final, Q_list, R_list = QR_algorithm(C, num_iterations=3)
    print(f"Step 4: QR algorithm completed with {len(Q_list)} iterations")

    # QR Demo on small matrix
    demo_on_small_matrix()
    print("Step 4 Demo: QR on 4x4 matrix completed")

    # Step 5: Eigen Decomposition
    eigenvalues, eigenvectors = eigen_decomposition(C)
    print(f"Step 5: Eigen decomposition completed - {len(eigenvalues)} eigenvalues")

    # Step 6: Explained Variance
    explained_variance_ratio, cumulative_variance, k_90 = explained_variance(eigenvalues)
    print(f"Step 6: 90% variance preserved with {k_90} components")

    # Step 7: Dimensionality Reduction
    k = 10
    W, T = dimension_reduction(X_centered, eigenvectors, k)
    print(f"Step 7: Reduced dimensions from {X_centered.shape[1]} to {k}")

    # Step 8: 2D Visualization
    T2 = visualize_2D(X_centered, eigenvectors, y)
    print("Step 8: 2D visualization completed")

    # Step 9: Reconstruction Error
    errors = reconstruction_error(X, X_centered, eigenvectors, mean_vector, y)
    print("Step 9: Reconstruction error analysis completed")

    # Step 10: m < n Case
    eigenvalues_subset, eigenvectors_subset, X_centered_subset = m_less_than_n(X, y)
    print("Step 10: m < n case analysis completed")

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


if __name__ == "__main__":
    main()