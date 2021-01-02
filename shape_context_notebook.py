"""
Reference:
    https://www.youtube.com/watch?v=m3rK3gx0tZo
    https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sc_digits.html
    https://kyamagu.github.io/mexopencv/matlab/ShapeContextDistanceExtractor.html
    https://deeplearning.lipingyang.org/shape-context-resources/
    https://github.com/creotiv/Python-Shape-Context
    https://medium.com/machine-learning-world/shape-context-descriptor-and-fast-characters-recognition-c031eac726f9
    https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/belongie-pami02.pdf
Visualization:
    https://www.youtube.com/watch?v=qJieWnkl9gQ

"""

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
from IPython import display
import collections
import time

from core.utils import *
from core.shape_context import ShapeContext

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
    # show_img(mm)
    return contours, hierarchy

def process_contour_actual(ori_img, threshold=0.92):
    # Expect an unprocessed cv2 img 2d. 1 channels. mask
    img = np.asarray(ori_img) / 255.0
    img[img < threshold] = 0
    img[img > 0] = 1
    img *= 255.0
    img = img.astype('uint8')
    # print(img.shape)
    # show_img(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mm = cv2.drawContours(tf.zeros_like(img).numpy(), contours, -1, (255,255,255), 3)
    # show_img(mm)
    return contours, hierarchy

def mask_background(img, threshold=0.992):
    # Expect an unprocessed cv2 img 3d. 3 channels. Image of the cloth. BGR
    origray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    origray = np.asarray(origray)
    origray = origray.astype('float32') / 255.0
    # show_img(origray)
    origray[origray > threshold] = 0
    origray *= 255.0
    origray = origray.astype(np.uint8)
    return origray

def draw_point(img, points):
    # Expect np.ndarray img shape h x w x 3. Range 0 - 255
    mx = np.max(img)
    new_img = img.copy()
    zeros = np.zeros(new_img.shape)
    for [x_i, y_i] in points:
        if mx > 1:
            new_img[y_i-1:y_i+1, x_i-1:x_i+1, :] = (255, 0, 0)
            zeros[y_i-1:y_i+1, x_i-1:x_i+1, :] = (255, 0, 0)
        else:
            new_img[y_i-1:y_i+1, x_i-1:x_i+1, :] = 1
            zeros[y_i-1:y_i+1, x_i-1:x_i+1, :] = 1
    return zeros, new_img

def get_points(cnts, simpleto=100):
    """

    Args:
        cnts ([type]): [description]
        simpleto (int, optional): [description]. Defaults to 100.

    Returns:
        np.ndarray: array points. shape(simpleto, 2)
    """
    # Expect contours got from cv2 contours
    mx = -1
    i_mx = -1

    for i, cnt in enumerate(cnts):
        if cnt.shape[0] > mx:
            i_mx = i
            mx = cnt.shape[0]
    # mx representing the maximum number of points in each vector in 
    # the contour matrix.
    # step = (mx // simpleto) + 1
    step = mx // simpleto

    points = np.asarray(cnts[i_mx]).reshape(-1, 2)
    # points = [points[i % points.shape[0]].tolist() for i in range(0, points.shape[0] + simpleto, step)][:simpleto]
    points = [points[i].tolist() for i in range(0, points.shape[0], step)][:simpleto]
    return np.asarray(points)

# Pipeline methods:
def compute_indices(cnts_cloth, cnts_pred):
    points_cloth = None
    points_pred = None
    min_cost = 1e8
    indexes = None
    init_source_id = None
    init_target_id = None
    for ori_n_points in range(40, 71, 10):
        for act_n_points in range(50, 71, 10):
            points_cloth_tmp = get_points(cnts_cloth, simpleto=ori_n_points) # 100 x 2
            points_pred_tmp = get_points(cnts_pred, simpleto=act_n_points) # 100 x 2

            descriptor1 = computer.compute(points_cloth_tmp) # 100 x 60
            descriptor2 = computer.compute(points_pred_tmp) # 100 x 60

            total, indexes_tmp, source_tmp_id, target_tmp_id = \
                computer.diff(descriptor1, descriptor2)
            # print(total)

            if total < min_cost:
                min_cost = total
                points_cloth = points_cloth_tmp.copy()
                points_pred = points_pred_tmp.copy()
                # indexes = indexes_tmp.copy()
                init_source_id = source_tmp_id.copy()
                init_target_id = target_tmp_id.copy()
    return min_cost, points_cloth, points_pred, init_source_id, init_target_id

def get_match_points(
    points_cloth, 
    points_pred, 
    init_source_id, 
    init_target_id, 
    dropout_rate=0.2,
    verbose=False):

    cnt = 0

    hs = collections.defaultdict(int)
    source = []
    target = []
    for p1, p2 in zip(init_source_id, init_target_id):
        if np.random.random() < dropout_rate:
            continue
        source.append(points_cloth[p1].tolist())
        target.append(points_pred[p2].tolist())
        hs[p1] += 1
        hs[p2] += 1
        cnt += 1
    if verbose:
        print(f"Number of matching pairs of points: {cnt}")

        for k, v in hs.items():
            if v > 2:
                print(k)
        print(f"Number of unique indices: {len(hs.keys())}")

    return source, target

# Plot matchh
def plot_match(computer: ShapeContext, cnts_cloth, cnts_pred):

    min_cost, points_cloth, points_pred, init_source_id, init_target_id = \
        compute_indices(cnts_cloth, cnts_pred)

    points1 = points_cloth
    x1 = [p[1] for p in points1]
    y1 = [p[0] for p in points1]
    points2 = (np.array(points_pred)+50).tolist()
    x2 = [p[1] for p in points2]
    y2 = [p[0] for p in points2]

    lines = []
    for p,q in zip(init_source_id, init_target_id):
        lines.append(((points1[p][1],points1[p][0]), (points2[q][1],points2[q][0])))
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(121)
    plt.gca().invert_yaxis()
    plt.plot(y1,x1,'go', y2,x2, 'ro')

    ax = plt.subplot(122)
    plt.gca().invert_yaxis()
    plt.plot(y1,x1,'go',y2,x2,'ro')
    for p1,p2 in lines:   
        plt.gca().invert_yaxis()
        plt.plot((p1[1],p2[1]),(p1[0],p2[0]), 'k-')
    plt.show()
    # print("Cosine diff: {}".format(cosine(P.flatten(), Q.flatten())))
    # print("Standard diff: {}".format(standard_cost))

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
cnts, hierarchy = process_contour_cloth(test)
# mm = cv2.drawContours(tf.zeros_like(test).numpy(), cnts, -1, (255,255,255), 3)
# plt.imshow(mm)
contours[8].shape
hierarchy.shape

# %%

# test for mask clothes

# cloth_mask_path = "./dataset/lip_mpv_dataset/preprocessed/clothing_mask/ZX121D004/ZX121D004-C11@10=cloth_front.jpg"
# mask_path = "./dataset/lip_mpv_dataset/MPV_192_256/ZX121D004/ZX121D004-C11@10=cloth_front.jpg"

cloth_mask_path = "./dataset/lip_mpv_dataset/preprocessed/clothing_mask/ZX121EA0A/ZX121EA0A-N11@10=cloth_front.jpg"
mask_path = "./dataset/lip_mpv_dataset/MPV_192_256/ZX121EA0A/ZX121EA0A-N11@10=cloth_front.jpg"

cloth = cv2.imread(mask_path, 1)
cnts_cloth, hierarchy = process_contour_cloth(cloth)

img = cv2.imread(cloth_mask_path, 0)
cnts_pred, hierarchy = process_contour_actual(img)

computer = ShapeContext(r_outer=6.0)

# %%

min_cost, points_cloth, points_pred, init_source_id, init_target_id = \
    compute_indices(cnts_cloth, cnts_pred)

# %%

# Draw points
points_only, abc = draw_point(np.asarray(mask_background(cloth)), points_cloth)
show_img(points_only / 255.0)

# Draw points
points_only, abc = draw_point(np.asarray(mask_background(img)), points_pred)
show_img(points_only / 255.0)

# %%
# getting the zipped source, target to 2 different array and count points

source, target = get_match_points(points_cloth, points_pred, init_source_id, init_target_id, verbose=True)

# %%

# See matching point

for i, ([sx, sy], [tx, ty]) in enumerate(zip(source, target)):
    display.clear_output(wait=True)
    print(f"Min cost: {min_cost}")
    print(f"[{sx}, {sy}] -- [{tx}, {ty}]")

    # Draw points
    points_only, abc = draw_point(np.asarray(mask_background(cloth)), source[:i + 1])
    show_img(points_only / 255.0)

    # Draw points
    points_only, abc = draw_point(np.asarray(mask_background(img)), target[:i + 1])
    show_img(points_only / 255.0)

    time.sleep(0.01)

# %%

plot_match(computer, cnts_cloth, cnts_pred)

# %%

tps = cv2.createThinPlateSplineShapeTransformer()

# Cartesian metrics
source = np.asarray(source, dtype=np.int32)
target = np.asarray(target, dtype=np.int32)

# source = np.asarray([[0, 0], [50, 50], [256, 33]], dtype=np.int32) # 256 x 192
# target = np.asarray([[5, 5], [45, 50], [256, 33]], dtype=np.int32) # 256 x 192

source = source.reshape(-1, len(source),2)
target = target.reshape(-1, len(target),2)

matches = list()
for i in range(0,len(source[0])):
    matches.append(cv2.DMatch(i,i,0))
tps.setRegularizationParameter(beta=0.9)
tps.estimateTransformation(target, source, matches)
# ret, tshape  = tps.applyTransformation (source)

# %%
prep_cloth = mask_background(cloth)
new_img = tps.warpImage(prep_cloth)

# %%

show_img(cloth)
show_img(prep_cloth)
show_img(img)
show_img(new_img)


# %%

# Pipeline !
# import tensorflow as tf
from tqdm import tqdm
from core.lip_dataset import train_generator, \
    IMG_SHAPE, \
    BATCH_SIZE, \
    TRAIN_PATH as LIP_TRAIN, \
    TEST_PATH as LIP_TEST, \
    DATASET_PATH as LIP_PATH, \
    DATASET_SRC as ORIGINAL_CLOTHS_FOLDER

computer = ShapeContext(r_outer=6.0)

def pipeline_step(original_cloth_path, actual_cloth_path):
    """[summary]

    Args:
        original_cloth_path ([type]): [description]
        actual_cloth_path ([type]): [description]

    Returns:
        cv image: shape(height, width, 3). BGR. Range 0-255. uint8
    """
    # Read image
    cloth = cv2.imread(original_cloth_path, 1)
    cnts_cloth, hierarchy = process_contour_cloth(cloth)

    img = cv2.imread(actual_cloth_path, 0)
    cnts_pred, hierarchy = process_contour_actual(img)

    min_cost, points_cloth, points_pred, init_source_id, init_target_id = \
        compute_indices(cnts_cloth, cnts_pred)

    source, target = \
        get_match_points(points_cloth, points_pred, init_source_id, init_target_id)

    # Cartesian metrics
    tps = cv2.createThinPlateSplineShapeTransformer()

    source = np.asarray(source, dtype=np.int32)
    target = np.asarray(target, dtype=np.int32)

    source = source.reshape(-1, len(source),2)
    target = target.reshape(-1, len(target),2)

    matches = list()
    for i in range(0,len(source[0])):
        matches.append(cv2.DMatch(i,i,0))

    tps.setRegularizationParameter(beta=0.5)
    tps.estimateTransformation(target, source, matches)
    
    prep_cloth = mask_background(cloth)
    new_img = tps.warpImage(prep_cloth)

    return new_img # some shape. range [0 - 255]

# %%

# Test pipeline
sample_actual_cloth_path = "./dataset/lip_mpv_dataset/preprocessed/clothing_mask/ZX121EA0A/ZX121EA0A-N11@10=cloth_front.jpg"
sample_original_cloth_path = "./dataset/lip_mpv_dataset/MPV_192_256/ZX121EA0A/ZX121EA0A-N11@10=cloth_front.jpg"
new_img = pipeline_step(sample_original_cloth_path, sample_actual_cloth_path)

# %%
new_img.shape
show_img(new_img)
# %%

# ORIGINAL_CLOTHS_FOLDER = get from lip_dataset.py
ACTUAL_CLOTHS_FOLDER = LIP_PATH / "preprocessed" / "clothing_mask"

OUT_FOLDER = LIP_PATH / "preprocessed" / "tps"
if not os.path.exists(OUT_FOLDER):
    os.mkdir(OUT_FOLDER)
r_str = r"\/.*\.jpg$"
# loop O(n^4)
# Loop through all cloth paths
with tqdm(total=LIP_TRAIN.shape[0]) as pbar:
    for p in LIP_TRAIN[:, 1]:
        try:
            if os.path.exists(str(OUT_FOLDER / p)):
                pbar.update(1)
                continue
            original_cloth_path = str(ORIGINAL_CLOTHS_FOLDER / p)
            actual_cloth_path = str(ACTUAL_CLOTHS_FOLDER / p)
            # new_img: cv image: shape(height, width, 3). BGR. Range 0-255. uint8
            new_img = pipeline_step(original_cloth_path, actual_cloth_path)
            # Write img to folder

            # Eg: img path is raw. I.e: does not contains the full path (dataset/lip_mpv_dataset/preprocessed)
            # img_path: RE321D05M\RE321D05M-Q11@8=person_half_front.jpg
            # img_path_name: RE321D05M-Q11@8=person_half_front.jpg
            # img_folder_name: RE321D05M
            img_path_name = re.findall(r_str, p)[0]
            img_folder_name = p[:(len(p) - len(img_path_name))]
            if not os.path.exists(OUT_FOLDER / img_folder_name):
                os.mkdir(OUT_FOLDER / img_folder_name)
            cv2.imwrite(str(OUT_FOLDER / p), new_img)
        except:
            pbar.update(1)
            continue
        pbar.update(1)
        
# %%
