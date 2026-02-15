import numpy as np

def tg_mds(D, n_components=1):
    """
    Torgerson-Guttman Multidimensional Scaling

    D: distance matrix (symmetric)
    """
    # Double-centering
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvecs[:, :n_components] * np.sqrt(eigvals[:n_components])