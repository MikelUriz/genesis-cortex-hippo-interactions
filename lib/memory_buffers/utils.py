from typing import List
import numpy as np


def l2normalize(x, dim=0):
    return x/np.linalg.norm(x, axis=dim, keepdims=True)


def combine_embeddings(x, y, theta):
    return np.hstack([theta*x, (1-theta)*y])


def generate_vector_by_similarity(x: List[float], cosine_similarity: float) -> List[float]:
    '''
    x: unit vector (L2-normalized), used as seed vector
    cosine_similarity: cos(theta), target distance between seed vector x and the new vector
    '''
    random_vector = np.random.randn(len(x))
    
    # create a vector perpendicular to x:
    x_orthogonal = random_vector - np.dot(random_vector, x)*x
    x_orthogonal = l2normalize(x_orthogonal)
    
    # y is the linear combination of x and x_orthogonal with coefficients cos(theta) and sin(theta), respectively:
    theta = np.arccos(cosine_similarity)
    y = np.cos(theta)*x + np.sin(theta)*x_orthogonal
    
    return y


def generate_vector_by_angle(x: List[float], theta: float) -> List[float]:
    '''
    x: unit vector (L2-normalized)
    theta: angle between x and the new vector
    '''
    random_vector = np.random.randn(len(x))
    
    # create a vector perpendicular to x:
    x_orthogonal = random_vector - np.dot(random_vector, x)*x
    x_orthogonal = l2normalize(x_orthogonal)
    
    # y is the linear combination of x and x_orthogonal with coefficients cos(theta) and sin(theta), respectively:
    y = np.cos(theta)*x + np.sin(theta)*x_orthogonal
    
    return y