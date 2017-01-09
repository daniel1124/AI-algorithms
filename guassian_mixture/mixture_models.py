from __future__ import division
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
import numpy as np
import scipy as sp
from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from helper_functions import image_to_matrix, matrix_to_image, flatten_image_matrix, unflatten_image_matrix, image_difference

from random import randint
from functools import reduce

def k_means_cluster(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """
    if len(image_values.shape) != 3:
        raise ValueError('Image shape should have height, width and depth.')
    
    _, width, _ = image_values.shape
    fimg = flatten_image_matrix(image_values)

    n, dim = fimg.shape
    
    if initial_means is None:
        means = fimg[np.random.choice(n, k, replace=False)]
    else:
        means = initial_means
    
    iter = 0
    max_iter = 100
    
    tol = 1e-10
    sse = 1e20
    dsse = 1e20 
        
    sse_matrix = np.zeros([len(fimg), k])
    
    while iter < max_iter and dsse > tol:
        
        for i in xrange(k):
            sse_matrix[:, i] = ((fimg - means[i]) ** 2).sum(axis=1)
        
        clusters = np.argmin(sse_matrix, axis=1).astype('int')
        sse_min = np.min(sse_matrix, axis=1)
        
        new_sse = sse_min.sum()
        
        means = np.array([fimg[clusters == i].mean(axis=0) for i in range(k)])
        
        dsse = np.abs(sse - new_sse)
        sse = new_sse
        #print('sse', sse)
        # print('dsse', dsse)
        iter += 1
    
    res_img = unflatten_image_matrix(means[clusters], width) 
    matrix_to_image(res_img, 'images/test_bird{}.png'.format(k))
    return res_img

def default_convergence(prev_likelihood, new_likelihood, conv_ctr, conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr+=1
    else:
        conv_ctr =0

    return conv_ctr, conv_ctr > conv_ctr_cap

from random import randint
import math
from scipy.misc import logsumexp

class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.image_matrix = image_matrix
        self.num_components = num_components
        if(means is None):
            self.means = [0]*num_components
        else:
            self.means = means
        self.variances = [0]*num_components
        self.mixing_coefficients = [0]*num_components

    def joint_prob(self, val):
        """Calculate the joint
        log probability of a greyscale
        value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """
        return logsumexp(np.log(self.mixing_coefficients) -
                         0.5 * np.log(2 * np.pi * self.variances) -
                         (val - self.means) ** 2 / (2 * self.variances))
   

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before
        train_model() in order for tests
        to execute correctly.
        """
        self.fimg = flatten_image_matrix(self.image_matrix)
        self.fimgs = self.fimg.squeeze()
        self.means = np.random.choice(self.fimgs, self.num_components, replace=False)
        self.variances = np.ones(self.num_components)
        self.mixing_coefficients = np.ones(self.num_components) / self.num_components
     

    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function that returns True if convergence is reached
        """
        conv_ctr = 0
        conv_flag = False
        prev_likelihood = 1e-20
                
        n = len(self.fimgs)
        
        while not conv_flag:
            
            self.temploglik = np.zeros([n, self.num_components])
            
            for i in xrange(self.num_components):
                
                self.temploglik[:, i] = np.log(self.mixing_coefficients[i]) - \
                    0.5 * np.log(2 * np.pi * self.variances[i]) - \
                    (self.fimgs - self.means[i]) ** 2 / (2 * self.variances[i])
            
            self.templogliksum = logsumexp(self.temploglik, axis=1)
            
            gamma = np.exp(self.temploglik - self.templogliksum.reshape(n, 1))
            
            for i in xrange(self.num_components):
                
                gi = gamma[:, i]
                self.means[i] = np.average(self.fimgs, weights=gi)
                self.variances[i] = np.average((self.fimgs - self.means[i]) ** 2,
                                               weights=gi)
                self.mixing_coefficients[i] = gi.sum() / n
            new_likelihood = self.likelihood()
            conv_ctr, conv_flag = convergence_function(prev_likelihood, new_likelihood,
                                                       conv_ctr)
            prev_likelihood = new_likelihood

    def segment(self):
        """
        Using the trained model,
        segment the image matrix into
        the pre-specified number of
        components. Returns the original
        image matrix with the each
        pixel's intensity replaced
        with its max-likelihood
        component mean.

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        _, width = self.image_matrix.shape
        seg = self.means[np.argmax(self.temploglik, axis=1)]
        
        return unflatten_image_matrix(seg, width)

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N),ln(sum((k=1 to K), mixing_k * N(x_n | mean_k, stdev_k) )))

        returns:
        log_likelihood = float [0,1]
        """
        if hasattr(self, 'templogliksum'):
            return self.templogliksum.sum()
        else:
            self.fimgs = flatten_image_matrix(self.image_matrix).squeeze()
            temploglik = np.zeros([len(self.fimgs), self.num_components])
            for i in xrange(self.num_components):
                
                temploglik[:, i] = np.log(self.mixing_coefficients[i]) - \
                    0.5 * np.log(2 * np.pi * self.variances[i]) - \
                    (self.fimgs - self.means[i]) ** 2 / (2 * self.variances[i])
            
            return logsumexp(temploglik, axis=1).sum()

    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly
        training the model and
        calculating its likelihood.
        Return the segment with the
        highest likelihood.

        params:
        iters = int

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        loglik = float('-inf')
        
        for i in xrange(iters):
            self.initialize_training()
            self.train_model()
            
            new_loglik = self.likelihood()
            
            if new_loglik > loglik:
                loglik = new_loglik
                best_means = np.copy(self.means)
                best_var = np.copy(self.variances)
                best_mix = np.copy(self.mixing_coefficients)
                best_seg = self.means[np.argmax(self.temploglik, axis=1)]

        self.means = best_means
        self.variances = best_var
        self.mixing_coefficients = best_mix
        
        _, width = self.image_matrix.shape
        
        return unflatten_image_matrix(best_seg, width)

def recur_dis(pre_mean, pre_dis, n, new_x):
    return pre_dis + 1.0 * (n - 1) / n * (new_x - pre_mean) ** 2

def ckmeans(x, k):
    
    x_ = np.sort(x)
    
    n = len(x_)
    d = [float('inf')] * (n + 1)
    d[0] = 0
    B = [[0] * k for i in range(n)]
    
    for m in range(1, k + 1):
        new_d = [float('inf')] * (n + 1)
        for i in range(1, n + 1):
            
            pre_dis = 0
            pre_mean = 0

            for j in range(i, 0, -1):
                
                dis = recur_dis(pre_mean, pre_dis, i - j + 1, x_[j - 1]) # d(xj ... xi)
                if d[j - 1] + dis < new_d[i]:
                    new_d[i] = d[j - 1] + dis
                    B[i - 1][m - 1] = j - 1

                pre_dis = dis
                pre_mean = (pre_mean * (i - j) + x_[j - 1]) / (i - j + 1)
                
        d = new_d
    
    res = []
    i = n - 1
    for j in range(k - 1, -1, -1):
        i = B[i][j]
        res.append(x_[i])
        i -= 1
    return np.array(list(reversed(res)))

class GaussianMixtureModelImproved(GaussianMixtureModel):
    """A Gaussian mixture model
    for a provided grayscale image,
    with improved training
    performance."""

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient initializations too if that works well.]
        """
        self.fimg = flatten_image_matrix(self.image_matrix)
        self.fimgs = self.fimg.squeeze()
        self.means = ckmeans(self.fimgs, self.num_components)

        self.variances = np.ones(self.num_components)
        self.mixing_coefficients = np.ones(self.num_components) / self.num_components
