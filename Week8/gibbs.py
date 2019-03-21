#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 13:02:28 2018

@author: paris
"""

import numpy as np
import seaborn as sns


if __name__ == '__main__':
    mus = np.array([5, 5])
    sigmas = np.array([[1, .9], [.9, 1]])


    def p_x_given_y(y, mus, sigmas):
        mu = mus[0] + sigmas[1, 0] / sigmas[0, 0] * (y - mus[1])
        sigma = sigmas[0, 0] - sigmas[1, 0] / sigmas[1, 1] * sigmas[1, 0]
        return np.random.normal(mu, sigma)
    
    
    def p_y_given_x(x, mus, sigmas):
        mu = mus[1] + sigmas[0, 1] / sigmas[1, 1] * (x - mus[0])
        sigma = sigmas[1, 1] - sigmas[0, 1] / sigmas[0, 0] * sigmas[0, 1]
        return np.random.normal(mu, sigma)
    
    
    def gibbs_sampling(mus, sigmas, iter=10000):
        samples = np.zeros((iter, 2))
        y = np.random.rand() * 10
    
        for i in range(iter):
            x = p_x_given_y(y, mus, sigmas)
            y = p_y_given_x(x, mus, sigmas)
            samples[i, :] = [x, y]
    
        return samples

    burn_in = 1000
    samples = gibbs_sampling(mus, sigmas, iter=20000)
    sns.jointplot(samples[burn_in:, 0], samples[burn_in:, 1])