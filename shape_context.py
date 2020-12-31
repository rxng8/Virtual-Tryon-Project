# %%

"""
This code is taken and modified from this url:
    https://github.com/creotiv/computer_vision/blob/master/shape_context/shape_context.py
"""

import numpy as np
import cv2
import math
from scipy.spatial.distance import cdist, cosine
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import tensorflow as tf

from core.utils import *


class ShapeContext(object):

    def __init__(self, nbins_r=5, nbins_theta=12, r_inner=0.1250, r_outer=2.0):
        # number of radius zones
        self.nbins_r = nbins_r
        # number of angles zones
        self.nbins_theta = nbins_theta
        # maximum and minimum radius
        self.r_inner = r_inner
        self.r_outer = r_outer

    def _hungarian(self, cost_matrix):
        """
            Here we are solving task of getting similar points from two paths
            based on their cost matrixes. 
            This algorithm has difficulty O(n^3)
            return total modification cost, indexes of matched points
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total = cost_matrix[row_ind, col_ind].sum()
        indexes = zip(row_ind.tolist(), col_ind.tolist())
        return total, indexes

    def _cost(self, hi, hj):
        # Compute cost between two different points.
        cost = 0
        for k in range(self.nbins_theta * self.nbins_r):
            if (hi[k] + hj[k]):
                cost += ((hi[k] - hj[k])**2) / (hi[k] + hj[k])

        return cost * 0.5

    def cost_by_paper(self, P, Q, qlength=None):
        # Compute cost matrix between all pairs
        p, _ = P.shape
        p2, _ = Q.shape
        d = p2
        if qlength:
            d = qlength
        C = np.zeros((p, p2))
        for i in range(p):
            for j in range(p2):
                C[i, j] = self._cost(Q[j] / d, P[i] / p)

        return C

    def compute(self, points):
        """
          Here we are computing shape context descriptor
        """
        t_points = len(points)
        # getting euclidian distance
        r_array = cdist(points, points)
        # getting two points with maximum distance to norm angle by them
        # this is needed for rotation invariant feature
        am = r_array.argmax()
        max_points = [am // t_points, am % t_points]
        # normalizing
        r_array_n = r_array / r_array.mean()
        # create log space
        r_bin_edges = np.logspace(np.log10(self.r_inner), np.log10(self.r_outer), self.nbins_r)
        r_array_q = np.zeros((t_points, t_points), dtype=int)
        # summing occurences in different log space intervals
        # logspace = [0.1250, 0.2500, 0.5000, 1.0000, 2.0000]
        # 0    1.3 -> 1 0 -> 2 0 -> 3 0 -> 4 0 -> 5 1
        # 0.43  0     0 1    0 2    1 3    2 4    3 5
        for m in range(self.nbins_r):
            r_array_q += (r_array_n < r_bin_edges[m])

        fz = r_array_q > 0

        # getting angles in radians
        theta_array = cdist(points, points, lambda u, v: math.atan2((v[1] - u[1]), (v[0] - u[0])))
        norm_angle = theta_array[max_points[0], max_points[1]]
        # making angles matrix rotation invariant
        theta_array = (theta_array - norm_angle * (np.ones((t_points, t_points)) - np.identity(t_points)))
        # removing all very small values because of float operation
        theta_array[np.abs(theta_array) < 1e-7] = 0

        # 2Pi shifted because we need angels in [0,2Pi]
        theta_array_2 = theta_array + 2 * math.pi * (theta_array < 0)
        # Simple Quantization
        theta_array_q = (1 + np.floor(theta_array_2 / (2 * math.pi / self.nbins_theta))).astype(int)

        # building point descriptor based on angle and distance
        nbins = self.nbins_theta * self.nbins_r
        descriptor = np.zeros((t_points, nbins))
        for i in range(t_points):
            sn = np.zeros((self.nbins_r, self.nbins_theta))
            for j in range(t_points):
                if (fz[i, j]):
                    sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
            descriptor[i] = sn.reshape(nbins)

        return descriptor

    def cosine_diff(self, P, Q):
        """
            Fast cosine diff.
        """
        P = P.flatten()
        Q = Q.flatten()
        assert len(P) == len(Q), 'number of descriptors should be the same'
        return cosine(P, Q)

    def diff(self, P, Q, qlength=None):
        """
            More precise but not very speed efficient diff.
            if Q is generalized shape context then it compute shape match.
            if Q is r point representative shape contexts and qlength set to 
            the number of points in Q then it compute fast shape match.
        """
        result = None
        C = self.cost_by_paper(P, Q, qlength)

        result = self._hungarian(C)

        return result

    @classmethod
    def tests(cls):
        # basics tests to see that all algorithm invariants options are working fine
        self = cls()

        def test_move():
            p1 = np.array([
                [0, 100],
                [200, 60],
                [350, 220],
                [370, 100],
                [70, 300],
            ])
            # +30 by x
            p2 = np.array([
                [0, 130],
                [200, 90],
                [350, 250],
                [370, 130],
                [70, 330]
            ])
            c1 = self.compute(p1)
            c2 = self.compute(p2)
            assert (np.abs(c1.flatten() - c2.flatten())
                    ).sum() == 0, "Moving points in 2d space should give same shape context vector"

        def test_scale():
            p1 = np.array([
                [0, 100],
                [200, 60],
                [350, 220],
                [370, 100],
                [70, 300],
            ])
            # 2x scaling
            p2 = np.array([
                [0, 200],
                [400, 120],
                [700, 440],
                [740, 200],
                [149, 600]
            ])
            c1 = self.compute(p1)
            c2 = self.compute(p2)
            assert (np.abs(c1.flatten() - c2.flatten())
                    ).sum() == 0, "Scaling points in 2d space should give same shape context vector"

        def test_rotation():
            p1 = np.array(
                [(144, 196), (220, 216), (330, 208)]
            )
            # 90 degree rotation
            theta = np.radians(90)
            c, s = np.cos(theta), np.sin(theta)
            R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
            p2 = np.dot(p1, R).tolist()

            c1 = self.compute(p1)
            c2 = self.compute(p2)
            assert (np.abs(c1.flatten() - c2.flatten())
                    ).sum() == 0, "Rotating points in 2d space should give same shape context vector"

        test_move()
        test_scale()
        test_rotation()
        print ('Tests PASSED')

# if __name__ == "__main__":
#     ShapeContext.tests()

# %%

def process_contour_cloth(img, threshold=0.992):
    # Expect an unprocessed cv2 img 3d. 3 channels. Image of the cloth
    origray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    origray = np.asarray(origray)
    origray = origray.astype('float32') / 255.0
    # show_img(origray)
    origray[origray > threshold] = 0 
    origray[origray > 0] = 1
    origray *= 255.0
    origray = origray.astype(np.uint8)
    # show_img(origray)
    ret, thresh = cv2.threshold(origray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mm = cv2.drawContours(tf.zeros_like(img).numpy(), contours, -1, (255,255,255), 3)
    show_img(mm)
    return contours, hierarchy

def process_contour_actual(ori_img, threshold=0.92):
    # Expect an unprocessed cv2 img 2d. 1 channels. mask
    img = np.asarray(ori_img) / 255.0
    img[img < threshold] = 0
    img[img > 0] = 1
    img *= 255.0
    img = img.astype('uint8')
    print(img.shape)
    # show_img(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mm = cv2.drawContours(tf.zeros_like(img).numpy(), contours, -1, (255,255,255), 3)
    show_img(mm)
    return contours, hierarchy



# %%
# Test for original clothes

lip_test = "./dataset/lip_mpv_dataset/MPV_192_256/0VB21E007/0VB21E007-T11@10=cloth_front.jpg"
orig_mask = "./dataset/lip_mpv_dataset/MPV_192_256/ZX121DA01/ZX121DA01-A11@12.1=cloth_front.jpg"

ori = cv2.imread(orig_mask)
origray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
origray = np.asarray(origray)
origray = origray.astype('float32') / 255.0
show_img(origray)
origray[origray > 0.992] = 0 
origray[origray > 0] = 1
origray *= 255.0
origray = origray.astype(np.uint8)
show_img(origray)

ret, thresh = cv2.threshold(origray, 127, 255, 0)
contours, hierarchy = cv2.findContours(origray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

mm = cv2.drawContours(tf.zeros_like(origray).numpy(), contours, -1, (255,255,255), 3)
plt.imshow(mm)

# %%
test = cv2.imread(orig_mask, 1)
cnts, hierarchy = process_contour(test)
# mm = cv2.drawContours(tf.zeros_like(test).numpy(), cnts, -1, (255,255,255), 3)
# plt.imshow(mm)
# %%

contours[8].shape

# %%

hierarchy.shape


# %%

# test for mask clothes

cloth_mask_path = "./dataset/lip_mpv_dataset/preprocessed/clothing_mask/ZX121D004/ZX121D004-C11@10=cloth_front.jpg"
mask_path = "./dataset/lip_mpv_dataset/MPV_192_256/ZX121D004/ZX121D004-C11@10=cloth_front.jpg"

cloth = cv2.imread(mask_path, 1)
cnts_cloth, hierarchy = process_contour_cloth(cloth)

img = cv2.imread(cloth_mask_path, 0)
cnts_pred, hierarchy = process_contour_actual(img)

# %%
def get_points(cnts, simpleto=100):
    # Expect contours got from cv2 contours
    mx = -1
    i_mx = -1

    for i, cnt in enumerate(cnts):
        if cnt.shape[0] > mx:
            i_mx = i
            mx = cnt.shape[0]

    step = mx // simpleto

    points = np.asarray(cnts[i_mx]).reshape(-1, 2)
    points = [points[i].tolist() for i in range(0, points.shape[0], step)][:simpleto]
    return np.asarray(points)

# %%

points_cloth = get_points(cnts_cloth)
points_pred = get_points(cnts_pred)

# %%
computer = ShapeContext()
descriptor1 = computer.compute(points_cloth)
descriptor2 = computer.compute(points_pred)

# cost_matrix = computer.cost_by_paper(descriptor1, descriptor2)

total, indexes = computer.diff(descriptor1, descriptor2)


# %%
cnt = 0
import collections
hs = collections.defaultdict(int)
for p1, p2 in indexes:
    # print(f"{p1}, {p2}")
    hs[p1] += 1
    hs[p2] += 1
    cnt += 1

print(cnt)

for k, v in hs.items():
    if v > 2:
        print(k)
print(len(hs.keys()))





