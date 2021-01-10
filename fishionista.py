#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

# %%

from scipy.io import loadmat
annots = loadmat('./dataset/fashionista-v0.2.1/fashionista-v0.2.1/fashionista_v0.2.1.mat')
# %%

annots['truths'][0][1]


