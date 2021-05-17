import os
import cv2
import shutil
import numpy as np
from PIL import Image

from align import *

from multiprocessing.pool import Pool

sift = cv2.xfeatures2d.SIFT_create()

im_lst = './select_changjing/changjing1.txt'
output = './outputs/'
image_dir = './changjing/'

ratio = 0.3
iter_num = 2000
fit_pos_cnt_thresh = 30

with open(im_lst, 'r') as f:
    lines = f.readlines()
cnt = len(lines)

ori_ims = []
trans_ims = []

for i, l in enumerate(lines):
    line = l.strip().strip('\n')[13:]
    if '_' in line:
        sku = line.split('_')[0]
        or_im_path = sku + '.png'
        tr_im_path = sku + '_1.png'
        
        ori_ims.append(or_im_path)
        trans_ims.append(tr_im_path)
    if i % 10000 == 0:
        print('%d / %d' % (i, cnt))
    
# for or_im, tr_im in zip(ori_ims, trans_ims):
#     or_im_path = image_dir + or_im
#     tr_im_path = image_dir + tr_im
#     if not os.path.exists(or_im_path) or not os.path.exists(tr_im_path):
# #         print('not found.')
#         continue
#     target_im = np.array(Image.open(or_im_path))
#     source_im = np.array(Image.open(tr_im_path))
#     source_rgb = source_im[:, :, :3]
#     a = source_im[:, :, 3]
#     a3 = np.stack([a, a, a], axis=-1)
    
#     kp_s, desc_s = extract_sift(source_rgb)
#     kp_t, desc_t = extract_sift(target_im)
#     fit_pos = match_sift(desc_s, desc_t)
    
#     print(fit_pos.shape[0])
#     if fit_pos.shape[0] < fit_pos_cnt_thresh:
#         print(fit_pos.shape[0])
#         os.remove(or_im_path)
#         os.remove(tr_im_path)
#         print('not same image')
#         continue
    
#     m = affine_matrix(kp_s, kp_t, fit_pos)
#     merge, warp_rgb, source = warp_image(source_rgb, target_im, m)
#     am, warp_a, _  = warp_image(a3, target_im, m)
    
#     warp_a = warp_a[:, :, 0]
#     warp_a = np.expand_dims(warp_a, axis=-1)
#     merge = np.concatenate((warp_rgb, warp_a), axis=2)
    
#     sku = or_im.split('.')[0]
#     save_path = output + sku + '_2.png'
#     or_new_path = output + or_im
#     tr_new_path = output + tr_im
#     shutil.move(or_im_path, or_new_path)
#     shutil.move(tr_im_path, tr_new_path)
#     Image.fromarray(merge.astype(np.uint8)).save(save_path)
#     print('precess %s' % save_path)
    
    
def func(img_lst):
    [or_im, tr_im] = img_lst
    or_im_path = image_dir + or_im
    tr_im_path = image_dir + tr_im
    if not os.path.exists(or_im_path) or not os.path.exists(tr_im_path):
#         print('not found.')
        return
    target_im = np.array(Image.open(or_im_path))
    source_im = np.array(Image.open(tr_im_path))
    source_rgb = source_im[:, :, :3]
    a = source_im[:, :, 3]
    a3 = np.stack([a, a, a], axis=-1)
    
    kp_s, desc_s = extract_sift(source_rgb)
    kp_t, desc_t = extract_sift(target_im)
    fit_pos = match_sift(desc_s, desc_t)
    
#     print(fit_pos.shape[0])
    if fit_pos.shape[0] < fit_pos_cnt_thresh:
#         print(fit_pos.shape[0])
        os.remove(or_im_path)
        os.remove(tr_im_path)
#         print('not same image')
        return
    
    m = affine_matrix(kp_s, kp_t, fit_pos)
    merge, warp_rgb, source = warp_image(source_rgb, target_im, m)
    am, warp_a, _  = warp_image(a3, target_im, m)
    
    warp_a = warp_a[:, :, 0]
    warp_a = np.expand_dims(warp_a, axis=-1)
    merge = np.concatenate((warp_rgb, warp_a), axis=2)
    
    sku = or_im.split('.')[0]
    save_path = output + sku + '_2.png'
    or_new_path = output + or_im
    tr_new_path = output + tr_im
    shutil.move(or_im_path, or_new_path)
    shutil.move(tr_im_path, tr_new_path)
    Image.fromarray(merge.astype(np.uint8)).save(save_path)
    print('precess %s' % save_path)
    
img_list = zip(ori_ims, trans_ims)
with Pool(20) as pool:
    print('pool.')
    pool.map(func, img_list)
