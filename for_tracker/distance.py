from __future__ import division, print_function, absolute_import
import torch
from torch.nn import functional as F
from scipy.spatial import distance
from sklearn import metrics
import numpy as np


def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.

    Examples::
       >>> from torchreid import metrics
       >>> input1 = torch.rand(10, 2048)
       >>> input2 = torch.rand(100, 2048)
       >>> distmat = metrics.compute_distance_matrix(input1, input2)
       >>> distmat.size() # (10, 100)
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.dim()
    )
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    elif metric == 'mahalanobis': ## 이것도 추가
        distmat = mahalanobis_distance(input1, input2)
    elif metric == 'jaccard':
        distmat = jaccard_distance(input1, input2)
    elif metric == 'dice':
        distmat = dice_distance(input1, input2)
    elif metric == 'chebyshev':
        distmat = chebyshev_distance(input1, input2)
    elif metric == 'tanimoto':
        distmat = tanimoto_distance(input1, input2)
    elif metric == 'pearson':
        distmat = pearson_distance(input1, input2)
    elif metric == 'hellinger':
        distmat = hellinger_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = mat1 + mat2
    distmat.addmm_(input1, input2.t(), beta=1, alpha=-2)
    return distmat


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


##########여기 추가함
'''
def mahalanobis_distance(input1, input2):
    """Computes Mahalanobis distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    # Normalize the inputs
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)

    # Combine normalized input1 and input2 to calculate the covariance matrix
    combined = torch.cat([input1_normed, input2_normed], dim=0)
    cov_matrix = torch.cov(combined.T)
    inv_cov_matrix = torch.inverse(cov_matrix)
    
    m, n = input1_normed.size(0), input2_normed.size(0)
    distmat = torch.zeros((m, n))

    for i in range(m):
        for j in range(n):
            diff = input1_normed[i] - input2_normed[j]
            dist = torch.sqrt(torch.dot(torch.dot(diff, inv_cov_matrix), diff.T))
            distmat[i, j] = dist

    return distmat
'''

# def mahalanobis_distance(input1, input2):
#     """Computes Mahalanobis distance.

#     Args:
#         input1 (torch.Tensor): 2-D feature matrix.
#         input2 (torch.Tensor): 2-D feature matrix.

#     Returns:
#         torch.Tensor: distance matrix.
#     """
#     # Normalize the inputs
#     input1_normed = F.normalize(input1, p=2, dim=1)
#     input2_normed = F.normalize(input2, p=2, dim=1)

#     # Combine normalized input1 and input2 to calculate the covariance matrix
#     combined = torch.cat([input1_normed, input2_normed], dim=0)
#     cov_matrix = torch.cov(combined.T)
#     inv_cov_matrix = torch.inverse(cov_matrix)

#     # Compute pairwise Mahalanobis distance
#     diff = input1_normed.unsqueeze(1) - input2_normed.unsqueeze(0)
#     left_term = torch.matmul(diff, inv_cov_matrix)
#     distmat = torch.sqrt(torch.sum(left_term * diff, dim=2))

#     return distmat


def mahalanobis_distance(input1, input2):
    """Computes Mahalanobis distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    # Combine normalized input1 and input2 to calculate the covariance matrix
    combined = torch.cat([input1, input2], dim=0)
    cov_matrix = torch.cov(combined.T)
    inv_cov_matrix = torch.inverse(cov_matrix)

    # Compute pairwise Mahalanobis distance
    diff = input1.unsqueeze(1) - input2.unsqueeze(0)
    left_term = torch.matmul(diff, inv_cov_matrix)
    distmat = torch.sqrt(torch.sum(left_term * diff, dim=2))

    return distmat


def jaccard_distance(input1, input2):
    """Computes Jaccard distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    intersection = torch.min(input1.unsqueeze(1), input2.unsqueeze(0)).sum(2)
    union = torch.max(input1.unsqueeze(1), input2.unsqueeze(0)).sum(2)
    distmat = 1 - intersection / union
    return distmat


def dice_distance(input1, input2):
    """Computes Dice distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    intersection = 2 * torch.min(input1.unsqueeze(1), input2.unsqueeze(0)).sum(2)
    sum_vals = input1.unsqueeze(1).sum(2) + input2.unsqueeze(0).sum(2)
    distmat = 1 - intersection / sum_vals
    return distmat


def chebyshev_distance(input1, input2):
    """Computes Chebyshev distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    distmat = torch.max(torch.abs(input1.unsqueeze(1) - input2.unsqueeze(0)), dim=2)[0]
    return distmat


def tanimoto_distance(input1, input2):
    """Computes Tanimoto distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_np = input1.cpu().numpy()
    input2_np = input2.cpu().numpy()
    distmat = distance.cdist(input1_np, input2_np, metric='rogerstanimoto')
    return torch.tensor(distmat)


def pearson_distance(input1, input2):
    """Computes Pearson distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_np = input1.cpu().numpy()
    input2_np = input2.cpu().numpy()

    # Calculate the mean of each row
    input1_mean = input1_np.mean(axis=1, keepdims=True)
    input2_mean = input2_np.mean(axis=1, keepdims=True)

    # Subtract the mean from the original matrix
    input1_centered = input1_np - input1_mean
    input2_centered = input2_np - input2_mean

    # Calculate the Pearson correlation coefficient
    numerator = np.dot(input1_centered, input2_centered.T)
    denominator = np.sqrt(np.sum(input1_centered ** 2, axis=1, keepdims=True)) * np.sqrt(np.sum(input2_centered ** 2, axis=1, keepdims=True)).T

    corr_matrix = numerator / denominator
    distmat = 1 - corr_matrix

    return torch.tensor(distmat)


def hellinger_distance(input1, input2):
    """Computes Hellinger distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_np = input1.cpu().numpy()
    input2_np = input2.cpu().numpy()

    # Compute the Hellinger distance
    distmat = np.sqrt(0.5 * np.sum((np.sqrt(input1_np[:, np.newaxis]) - np.sqrt(input2_np)) ** 2, axis=2))

    return torch.tensor(distmat)