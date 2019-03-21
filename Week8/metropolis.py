#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:59:50 2018

@author: paris
"""

import numpy as np
import scipy.stats as st
import seaborn as sns


if __name__ == '__main__':
    
    mus = np.array([5, 5])
    sigmas = np.array([[1, .9], [.9, 1]])
    
    
    def circle(x, y):
        return (x-1)**2 + (y-2)**2 - 3**2
    
    
    def pgauss(x, y):
        return st.multivariate_normal.pdf([x, y], mean=mus, cov=sigmas)
    
    
    def metropolis(p, iter=1000):
        x, y = 0., 0.
        samples = np.zeros((iter, 2))
    
        for i in range(iter):
            x_star, y_star = np.array([x, y]) + np.random.normal(size=2)
            if np.random.rand() < p(x_star, y_star) / p(x, y):
                x, y = x_star, y_star
            samples[i] = np.array([x, y])
    
        return samples
    
    burn_in = 1000
    samples = metropolis(circle, iter=20000)
    sns.jointplot(samples[burn_in:, 0], samples[burn_in:, 1])

    samples = metropolis(pgauss, iter=20000)
    sns.jointplot(samples[burn_in:, 0], samples[burn_in:, 1])