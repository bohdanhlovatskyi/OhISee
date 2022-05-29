import numpy as np
import random

class EssentialMat:
    def __init__(self, matrix=None):
        if matrix is None:
            matrix = np.eye(3)
        assert matrix.shape == (3, 3) and 'Invalid shape of transformation matrix'
        self.params = matrix

    def estimate(self, pts1: np.array, pts2: np.array):
        """
        points are already normalized
        pts1[i] = [x1, y1]; pts2[i] = [x2, y2]

                    (e1,e2,e3)
        (x2,y2,1) * (e4,e5,e6) * (x1,y1,1)^T = 0
                    (e7,e8,e9)

        [x1x2, x1y2, x1, y1x2, y1y2, y1, x2, y2, 1] · e = 0
        Resulted `e` is the eigenvector of `A^T @ A` which correspond to smallest eigenvalue.

        :param pts1: (N, 2) np.array - coordinates on first (source) image
        :param pts2: (N, 2) np.array - coordinates on second (destination) image

        :return: True, if success
        """
        size = pts1.shape[0]
        assert 8 <= size == pts2.shape[0] and 'Can\'t find E with less than 4 points'

        x1, y1 = pts1.T
        x2, y2 = pts2.T
        mat = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, np.ones(len(x1))]).T

        # compute linear least square solution with ||e|| = 1
        val, vec = np.linalg.eig(mat.T @ mat)
        E = vec[:, -1]

        # set eigenvalues to [1,1,0]
        U, L, V_T = np.linalg.svd(E.reshape((3, 3)))
        L[0] = L[1] = (L[0] + L[1]) / 2.
        L[2] = 0
        E = U @ np.diag([1,1,0]) @ V_T

        self.params = E
        return True

    def residuals(self, pts1: np.array, pts2: np.array):
        """
        Compute the Sampson distance.
        src: https://stackoverflow.com/questions/26582960/sampson-error-for-five-point-essential-matrix-estimation
             https://arxiv.org/pdf/1706.07886.pdf

        The Sampson distance is the first approximation to the geometric error.

        :param pts1: (N, 2) np.array - coordinates on first (source) image
        :param pts2: (N, 2) np.array - coordinates on second (destination) image

        :return: (N,) np.array - Sampson distance.
        """
        pts1_ones = np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1)
        pts2_ones = np.concatenate([pts2, np.ones((pts2.shape[0], 1))], axis=1)

        F_pts1  = self.params   @ pts1_ones.T
        Ft_pts2 = self.params.T @ pts2_ones.T

        num = np.sum(pts2_ones*F_pts1.T, axis=1)
        denom = F_pts1[0]*F_pts1[0] + F_pts1[1]*F_pts1[1]* + Ft_pts2[0]*Ft_pts2[0]* + Ft_pts2[1]*Ft_pts2[1]

        return np.abs(num) / np.sqrt(denom)


def ransac(data: tuple or list, model_class, sample_size,
           residual_threshold=0.001, max_attempts=100, prob=1.):
    """
    RANSAC = RANdom SAmple Consensus algorithm

    steps:
      1) sample (randomly) points to fit the model
      2) calculate model parameters using chosen simple
      3) score by the fraction of inliers within
         a preset threshold of our model

    :param data: data to fit the model
    :param model_class: model, has following methods:
             * ``success = estimate(*data)`` - indicates whether the model estimation succeeded
             * ``residuals(*data)``
    :param sample_size: size of the sample in each run
    :param residual_threshold: threshold of accuracy of inliers
    :param max_attempts: number of maximum tries of running algorithm
    :param prob: probability of that ransac get correct result
    :return:
        :best_model: founded Essential matrix
        :inliers_mask: mask of founded inliers
    """

    def new_max_attempts(num_inliers, num_values, samples_size, prob_=1.):
        """ number trials such that at probability of all fine is `prob`. """
        nom = 1. - prob_
        inliers_ratio = num_inliers / num_values
        denom = 1 - np.power(inliers_ratio, samples_size)

        if num_inliers == 0. or nom == 0 or denom == 1:
            return np.inf
        elif denom == 0:
            return 0
        return int(np.ceil(np.log(nom) / np.log(denom)))

    size = len(data[0])
    assert 0 < sample_size < size and "`sample_size` must be in range (0, <number-of-samples>)"

    best_model = None
    best_inliers_num = 0
    best_inliers_residuals_sum = np.inf
    best_inliers = None

    done_inters = 0

    while done_inters < max_attempts:
        sample_ind = random.sample(range(0, size), sample_size)

        samples = [d[sample_ind] for d in data]

        sample_model = model_class()
        sample_model.estimate(*samples)
        sample_model_residuals = np.abs(sample_model.residuals(*data))
        sample_model_inliers = sample_model_residuals < residual_threshold
        sample_model_residuals_sum = np.sum(sample_model_residuals ** 2)
        sample_inliers_num = np.sum(sample_model_inliers)

        if (best_inliers_num < sample_inliers_num
                or (sample_inliers_num == best_inliers_num
                    and sample_model_residuals_sum < best_inliers_residuals_sum)):
            best_model = sample_model
            best_inliers_num = sample_inliers_num
            best_inliers_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            max_attempts = min(max_attempts,
                               new_max_attempts(best_inliers_num, size,
                                                sample_size, prob))

        done_inters += 1

    # select inliers for each data array
    if best_inliers is not None:
        data_inliers = [d[best_inliers] for d in data]
        best_model.estimate(*data_inliers)

    return best_model, best_inliers

