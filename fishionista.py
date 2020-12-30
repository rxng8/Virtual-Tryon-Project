# %%

from scipy.io import loadmat
annots = loadmat('./dataset/fashionista-v0.2.1/fashionista-v0.2.1/fashionista_v0.2.1.mat')
# %%

annots['truths'][0][1]


