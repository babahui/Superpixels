"""
=========================
image segmentation method
=========================

for image segmentation:

"""

print(__doc__)

import numpy as np
import scipy
from skimage.segmentation import slic, felzenszwalb
from skimage.future import graph
import networkx as nx
from scipy import sparse
import matplotlib
import matplotlib.pyplot as plt
from admm_algorithms import admm, relation_density_admm, var_admm

from copy import deepcopy
import math

from mpl_toolkits.mplot3d import Axes3D
from skimage.future import graph as gg
from skimage import data, segmentation, color

from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.io import imsave

import pylab as pl
from os.path import join
import os

from numpy import linalg as LA
from sklearn.preprocessing import normalize
from imageio import imread

from scipy.spatial import distance
from skimage.feature.texture import local_binary_pattern
import time


def normal_image_segmentation(m_img, sp_mode="similarity", sp_connectivity=4, num_cuts=2, n_iter=1000, graph_met='lib_met'):
    # read image
#     base_name = os.getcwd()
#     m_img = scipy.misc.imread(join(base_name, img_name))
    
    # do the superpixels method
    m_img = np.asarray(m_img)
    segments = np.zeros((m_img.shape[0], m_img.shape[1]))
    print(m_img.shape[0], m_img.shape[1])
    k = 1
    for i in range(m_img.shape[0]):
        for j in range(m_img.shape[1]):
            segments[i, j] = k
            k += 1
                
    
#     segments = slic(m_img, compactness=30, n_segments=sp_num)
#     segments = felzenszwalb(m_img, scale=50, sigma=0.5, min_size=100)


    # generate graph matrix
    import warnings

    if graph_met == 'lib_met':
        g = graph.rag_mean_color(m_img, segments, mode=sp_mode, connectivity=sp_connectivity)
        w = nx.to_scipy_sparse_matrix(g, format='csc')
        entries = w.sum(axis=0)
        d = sparse.dia_matrix((entries, 0), shape=w.shape).tocsc()
        m = w.shape[0]
        d2 = d.copy()
        d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)
        matrix = d2 * (d - w) * d2

        # matrix eigen-decomposition, scipy.sparse.linalg
        vals, vectors = scipy.sparse.linalg.eigsh(matrix, which='SM', k=min(100, m - 2))
        vals, vectors = np.real(vals), np.real(vectors)
        index1, index2, index3 = np.argsort(vals)[0], np.argsort(vals)[1], np.argsort(vals)[2]
        ev1, ev2, ev3 = vectors[:, index1], vectors[:, index2], vectors[:, index3]
        
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d.axes3d import Axes3D
#         # 3D plot
#         fig = plt.figure()
#         ax=Axes3D(fig, elev=-150, azim=110)
# #         ax.title("eigen-space projection")
# #         ax.scatter(v[:, 0], v[:, 1], v[:, 2])
#         ax.scatter(ev1, ev2, ev3)
#         plt.show()
        
    elif graph_met == 'syn_met':
        
        length = m_img.shape[0] * m_img.shape[1]
        graph = np.zeros((length, length))
        for i in range(length):
            for j in range(length):
                i_x, i_y = (i - i % m_img.shape[1]) / m_img.shape[1], i % m_img.shape[1] 
                j_x, j_y = (j - j % m_img.shape[1]) / m_img.shape[1], j % m_img.shape[1]
                
                if len(m_img.shape) == 2:
                    if abs(i_x - j_x) <= 4 and abs(i_y - j_y) <= 4:
#                         graph[i, j] = abs(m_img[i_x, i_y] - m_img[j_x, j_y])

                        diff = abs(m_img[i_x, i_y] - m_img[j_x, j_y])
                        graph[i, j] = math.exp(-(diff ** 2) / 1.0)

#                     graph[i, j] = abs(m_img[i_x, i_y] - m_img[j_x, j_y])


        # sparse eigenvectors
#         g = graph.rag_mean_color(m_img, segments, mode=sp_mode, connectivity=sp_connectivity)
        g = nx.Graph(graph)
        w = nx.to_scipy_sparse_matrix(g, format='csc')
        entries = w.sum(axis=0)
        d = sparse.dia_matrix((entries, 0), shape=w.shape).tocsc()
        m = w.shape[0]
        d2 = d.copy()
        d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)
        matrix = d2 * (d - w) * d2

        # matrix eigen-decomposition, scipy.sparse.linalg
        vals, vectors = scipy.sparse.linalg.eigsh(matrix, which='SM', k=min(100, m - 2))
        vals, vectors = np.real(vals), np.real(vectors)
        index1, index2, index3 = np.argsort(vals)[0], np.argsort(vals)[1], np.argsort(vals)[2]
        ev1, ev2, ev3 = vectors[:, index1], vectors[:, index2], vectors[:, index3]
        
        # density eigenvectors
#         vals, vectors = np.linalg.eigh(graph)
#         vals, vectors = np.real(vals), np.real(vectors)
#         index1, index2, index3 = np.argsort(vals)[0], np.argsort(vals)[1], np.argsort(vals)[2]
#         # index1, index2, index3 = np.argsort(vals)[::-1][0], np.argsort(vals)[::-1][1], np.argsort(vals)[::-1][2]
#         ev2, ev1, ev3 = vectors[:, index1], vectors[:, index2], vectors[:, index3]                    
        
    
#         ev1, ev2, ev3 = syn_graph_met(m_img, segments)
        
    else:
        warnings.warn('admm_met argument missing', UserWarning)
        
    print('graph success')
    
    # matrix = matrix.toarray()
    matplotlib.interactive(True)

    sp_label = admm(n_vector=ev2, n_iter=n_iter, num_cuts=num_cuts)
    sp_label = np.reshape(sp_label, (m_img.shape[0], m_img.shape[1]))
    
    return sp_label
    
#     if admm_met == 'admm':
#         sp_label = admm(n_vector=ev2, n_iter=n_iter, num_cuts=num_cuts)

#     elif admm_met == 'density_admm':
#         sp_label = relation_density_admm(n_vector=ev2, num_cuts=num_cuts, n_iter=n_iter, thres_cons=thres_cons)
        
#     elif admm_met == 'boundary_admm':
#         sp_label = bound_density_admm(n_vector=ev2, num_cuts=num_cuts, n_iter=n_iter, dens_coff=dens_coff)

#     else:
#         warnings.warn('admm_met argument missing', UserWarning)

       
    p_label, labels = pixels_label(m_img, segments, sp_label)
#     print(p_label.shape)
#     print(len(labels))
    
    #image_plot(m_img, p_label, labels)
#     return p_label, labels, ev1, ev2, ev3, sp_label


def large_image_seg(img_path="train/35070", sp_num=400, sp_mode="similarity", sp_connectivity=2, num_cuts=3, admm_met='admm', thres_cons=3, n_iter=1000, graph_met='lib_met', dens_coff=3, lambda_coff=None, merge=False, dist_hist=False):
    # read image
#     base_name = os.getcwd()
#     m_img = scipy.misc.imread(join(base_name, img_path))

    m_img = imread(img_path)
    
    # do the superpixels method
    segments = slic(m_img, compactness=30, n_segments=sp_num)
#     segments = felzenszwalb(m_img, scale=10, sigma=0.5, min_size=100)

    # generate graph matrix
    import warnings

    if graph_met == 'lib_met':
        g = graph.rag_mean_color(m_img, segments, mode=sp_mode, connectivity=sp_connectivity)
        w = nx.to_scipy_sparse_matrix(g, format='csc')
        entries = w.sum(axis=0)
        d = sparse.dia_matrix((entries, 0), shape=w.shape).tocsc()
        m = w.shape[0]
        d2 = d.copy()
        d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)
        matrix = d2 * (d - w) * d2

        # matrix eigen-decomposition, scipy.sparse.linalg
        vals, vectors = scipy.sparse.linalg.eigsh(matrix, which='SM', k=min(100, m - 2))
        vals, vectors = np.real(vals), np.real(vectors)
        index1, index2, index3 = np.argsort(vals)[0], np.argsort(vals)[1], np.argsort(vals)[2]
        ev1, ev2, ev3 = vectors[:, index1], vectors[:, index2], vectors[:, index3]
        
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d.axes3d import Axes3D
#         # 3D plot
#         fig = plt.figure()
#         ax=Axes3D(fig, elev=-150, azim=110)
# #         ax.title("eigen-space projection")
# #         ax.scatter(v[:, 0], v[:, 1], v[:, 2])
#         ax.scatter(ev1, ev2, ev3)
#         plt.show()
        
    if graph_met == 'syn_met':
        ev1, ev2, ev3 = syn_graph_met(m_img, segments, lambda_coff=lambda_coff, dist_hist=dist_hist)     
        print('syn_met passed')
        
    else:
        warnings.warn('admm_met argument missing', UserWarning)
        

    # matrix = matrix.toarray()
    matplotlib.interactive(True)

            
    if admm_met == 'admm':
        sp_label = admm(n_vector=ev2, n_iter=n_iter, num_cuts=num_cuts, merge=merge)

    elif admm_met == 'density_admm':
        sp_label = relation_density_admm(n_vector=ev2, num_cuts=num_cuts, n_iter=n_iter, thres_cons=thres_cons)
        
    elif admm_met == 'boundary_admm':
        sp_label = bound_density_admm(n_vector=ev2, num_cuts=num_cuts, n_iter=n_iter, dens_coff=dens_coff)
                
    elif admm_met == 'var_admm':
        sp_label = var_admm(n_vector=ev2, num_cuts=num_cuts, n_iter=n_iter)
        
    else:
        warnings.warn('admm_met argument missing', UserWarning)

        
    p_label, labels = pixels_label(m_img, segments, sp_label)
#     print(p_label.shape)
#     print(len(labels))
    
    #image_plot(m_img, p_label, labels)
#     return p_label, labels, ev1, ev2, ev3, sp_label
    return p_label

def pixels_label(m_img, segments, sp_label):
    # get superpixels position
    row, col = m_img.shape[0], m_img.shape[1]
    seg_label = []
    for i in range(row):
        for j in range(col):
            l = segments[i, j]
            if l not in seg_label:
                seg_label.append(l)
    sp_pos = []
    for i in seg_label:
        i_pos = []
        for m in range(row):
            for n in range(col):
                if segments[m, n] == i:
                    i_pos.append([m, n])
        sp_pos.append(i_pos)

    labels = []
    for i in range(len(sp_label)):
        if sp_label[i] not in labels:
            labels.append(sp_label[i])

    p_label = np.ones((row, col))
    for i in range(len(labels)):
        label_i_index = [j for j in range(len(sp_label)) if sp_label[j] == labels[i]]
        label_i_p = []
        for k in label_i_index:
            label_i_p.extend(sp_pos[k])
        color = int(i)
        for p in label_i_p:
            [cor_i, cor_j] = p
            p_label[cor_i, cor_j] = color

    p_label = np.asarray(p_label, dtype=int)

    return p_label, labels

def image_plot(m_img, p_label, labels):
    fig = plt.figure()
#     plt.imshow(m_img)
#     print("labels---------------------_", labels)

#     cmap = plt.cm.get_cmap("Spectral")
#     print('labels', labels)
#     for i in range(len(labels)):
# #         plt.contour(p_label==l, contour=1, colors=[plt.cm.spectral(l/float(len(labels)))])
#         plt.contour(p_label==labels[i], contour=1, colors=cmap(labels[i]/float(len(labels))))
#     plt.title("our method, distance=|R-R'|+|G-G'|+|B-B'|")

    # matplotlib.interactive(False)
    
    plt.imshow(mark_boundaries(m_img, p_label, color=(0, 0, 1)))
    plt.show()
    
    
def superpixels_generate(img_name="train/35070"):
    base_name = "/home/yy/berkeley_datasets/BSR/BSDS500/data/images/"
    m_img = scipy.misc.imread(base_name + img_name + ".jpg")
    segments = slic(m_img, compactness=30, n_segments=400)

    return m_img, segments

def superpixels_label(m_img, sp_mode="similarity", sp_connectivity=2, num_cuts=5):
    # generate graph matrix
    g = graph.rag_mean_color(m_img, segments, mode=sp_mode, connectivity=sp_connectivity)
    w = nx.to_scipy_sparse_matrix(g, format='csc')
    entries = w.sum(axis=0)
    d = sparse.dia_matrix((entries, 0), shape=w.shape).tocsc()
    m = w.shape[0]
    d2 = d.copy()
    d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)
    matrix = d2 * (d - w) * d2

    # matrix = matrix.toarray()
    matplotlib.interactive(True)

    # matrix eigen-decomposition, scipy.sparse.linalg
    vals, vectors = scipy.sparse.linalg.eigsh(matrix, which='SM', k=min(100, m - 2))
    vals, vectors = np.real(vals), np.real(vectors)
    index1, index2, index3 = np.argsort(vals)[0], np.argsort(vals)[1], np.argsort(vals)[2]
    ev1, ev2, ev3 = vectors[:, index1], vectors[:, index2], vectors[:, index3]

    sp_label = admm(n_vector=ev2, num_cuts=num_cuts)

    return sp_label

def pixels_label(m_img, segments, sp_label):
    # get superpixels position
    row, col = m_img.shape[0], m_img.shape[1]
    seg_label = []
    for i in range(row):
        for j in range(col):
            l = segments[i, j]
            if l not in seg_label:
                seg_label.append(l)
    sp_pos = []
    for i in seg_label:
        i_pos = []
        for m in range(row):
            for n in range(col):
                if segments[m, n] == i:
                    i_pos.append([m, n])
        sp_pos.append(i_pos)

    labels = []
    for i in range(len(sp_label)):
        if sp_label[i] not in labels:
            labels.append(sp_label[i])

    p_label = np.ones((row, col))
    for i in range(len(labels)):
        label_i_index = [j for j in range(len(sp_label)) if sp_label[j] == labels[i]]
        label_i_p = []
        for k in label_i_index:
            label_i_p.extend(sp_pos[k])
        color = int(i)
        for p in label_i_p:
            [cor_i, cor_j] = p
            p_label[cor_i, cor_j] = color

    p_label = np.asarray(p_label, dtype=int)

    return p_label, labels

    
def syn_graph_met(m_img, segments, lambda_coff, dist_hist=False):
    image = m_img

    # init graph by first method, by color distance metric between superpixels.
    row = image.shape[0]
    col = image.shape[1]
#     print(row, col)
    segmentsLabel = []
    for i in range(row):
        for j in range(col):
            l = segments[i, j]
            if l not in segmentsLabel:
                segmentsLabel.append(l)
    
    position = []
    ave_position = []
    flatten_position = []
    
    for i in segmentsLabel:
        pixel_position = []
        flatten_pos = []
        for m in range(row):
            for n in range(col):
                if segments[m, n] == i:
                    pixel_position.append([m, n])
                    flatten_pos.append(m * col + n)
                    
        position.append(pixel_position)
        flatten_position.append(flatten_pos)
        
        pixel_position = np.asarray(pixel_position)
        ave_position.append((sum(pixel_position) / len(pixel_position)).tolist())
        
    # generate average color value and red, green, blue color values
    average = []
    red_average = []
    green_average = []
    blue_average = []
    for i in range(len(position)):
        val = 0
        red_val = 0
        green_val = 0
        blue_val = 0
        for j in position[i]:
            [m, n] = j
            val += 0.299*image[m, n, 0] + 0.587*image[m, n, 1] + 0.114*image[m, n, 2]
            red_val += image[m, n, 0]
            green_val += image[m, n, 1]
            blue_val += image[m, n, 2]
            # val += image[m, n]
        average.append(val/len(position[i]))
        red_average.append(red_val/len(position[i]))
        green_average.append(green_val/len(position[i]))
        blue_average.append(blue_val/len(position[i]))

    # distance metric: by average value
    # average = []
    # for i in range(len(position)):
    #     val = 0
    #     for j in position[i]:
    #         [m, n] = j
    #         val += 0.299*image[m, n, 0] + 0.587*image[m, n, 1] + 0.114*image[m, n, 2]
    #         # val += image[m, n]
    #     average.append(val/len(position[i]))

    # length = len(position)
    # graph = np.zeros((length, length))
    # for i in range(length):
    #     for j in range(length):
    #         graph[i, j] = abs(average[i] - average[j]) ** 2
    
    graph_time = time.time()
    
    # fully connected
    sigma = 255.0
    length = len(position)
    graph = np.zeros((length, length))
    
    # settings for LBP
    radius = 2
    n_points = 8 * radius
    METHOD = 'uniform'
    
    img, lbp = [], []
    for i in range(3):
        c_img = image[:,:,i]
        c_lbp = local_binary_pattern(c_img, n_points, radius, METHOD)
        img.append(c_img)
        lbp.append(c_lbp)
    
    for i in range(length):
        for j in range(length):     
            if not dist_hist:
                diff = abs(red_average[i]-red_average[j]) + abs(green_average[i]-green_average[j]) + abs(blue_average[i]-blue_average[j])
                if lambda_coff:
                    dist = LA.norm(np.asarray(ave_position[i]) - np.asarray(ave_position[j]))
                    diff = diff + lambda_coff * dist 
            else:
                # reads an input image, color mode
                hist1 = hist(flatten_position[i], img, lbp)
                hist2 = hist(flatten_position[j], img, lbp)
                
                diff = abs(distance.cityblock(hist1, hist2))
                
            graph[i, j] = diff
            # graph[i, j] = math.e ** (-(diff ** 2) / sigma)

    print('graph construction time: ', time.time() - graph_time)    
    
    # matrix eigen-decomposition, scipy.sparse.linalg
    vals, vectors = np.linalg.eigh(graph)
    vals, vectors = np.real(vals), np.real(vectors)
    index1, index2, index3 = np.argsort(vals)[0], np.argsort(vals)[1], np.argsort(vals)[2]
    # index1, index2, index3 = np.argsort(vals)[::-1][0], np.argsort(vals)[::-1][1], np.argsort(vals)[::-1][2]
    ev1, ev2, ev3 = vectors[:, index1], vectors[:, index2], vectors[:, index3]
    

#     fig = plt.figure()
#     ax1 = Axes3D(fig, elev=-150, azim=110)
#     ax1.scatter(ev1, ev2, ev3, c=sp_label)
#     ax1.set_xlabel("x")
#     ax1.set_ylabel("y")
#     ax1.set_zlabel("z")
#     # plt.title("distance=|R-R'|+|G-G'|+|B-B'|")
#     plt.show()

    return ev1, ev2, ev3


def hist(position, img, lbp):
    
    # color hist: for each channel * 16 bins
    # find frequency of pixels in range 0-255, calculate histogram of blue, green or red channel respectively.
    color_hist = []
    for i in range(3):
        c_img = img[i]
        sp_arr = np.take(c_img, position)
                
        histr, bins = np.histogram(sp_arr, bins=np.linspace(0, 256, num=21))
#         histr = np.reshape(histr, (16, 16)) # 16 bins
#         histr = np.sum(histr, axis=1)  
        color_hist.append(histr)
        
    color_hist = normalize(color_hist).flatten()
    color_hist = np.asarray(color_hist)
    
    # texture hist, for each channel, orientation * 10 bins
    texture_hist = []
    for i in range(3):
        c_lbp = lbp[i]
        sp_lbp = np.take(c_lbp, position)
            
        n_bins = int(sp_lbp.max() + 1)
        histr, _ = np.histogram(sp_lbp, density=True, bins=np.linspace(0, n_bins, num=11))
        texture_hist.append(histr)
    
    texture_hist = normalize(texture_hist).flatten()
    texture_hist = np.asarray(texture_hist)   
        
    hist = np.append(color_hist, texture_hist)
#     print('-----hist--------', hist.shape)
    return np.append(color_hist, texture_hist)
    
if __name__ == "__main__":
    # image_list = ["2018", "3063", "5096", "6046", "8068", "10081", "14085", "14092", "15011", "15062"]
    # for image_name in image_list:
    #     m_img, segments = superpixels_generate(img_name="test/"+image_name)
    #     sp_label = superpixels_label_test(m_img=m_img, sp_connectivity=4, segments=segments, num_cuts=3)
    #     p_label, labels = pixels_label(m_img=m_img, segments=segments, sp_label=sp_label)
    #     # image_plot(m_img=m_img, p_label=p_label, labels=labels)
    #     # plt.show()

    #     seg_boundary = find_boundaries(p_label).astype(np.uint8)
    #     for i in range(0, seg_boundary.shape[0]):
    #         for j in range(0, seg_boundary.shape[1]):
    #             if seg_boundary[i, j] == 1:
    #                 seg_boundary[i, j] = 255
    #     path = "/home/yy/hust_lab/CV/github_spectral_clustering/reconstruct_project/segmentation_results/our_cut/" + image_name + ".png"
    #     imsave(path, seg_boundary, cmap='gray')
    #     image_plot(m_img=m_img, p_label=p_label, labels=labels)

    for num_cut in [3]:
        image_name = "15011"
        m_img, segments = superpixels_generate(img_name="test/"+image_name)
        sp_label = superpixels_label_test(m_img=m_img, sp_connectivity=4, segments=segments, num_cuts=num_cut)
        p_label, labels = pixels_label(m_img=m_img, segments=segments, sp_label=sp_label)

        seg_boundary = find_boundaries(p_label).astype(np.uint8)
        for i in range(0, seg_boundary.shape[0]):
            for j in range(0, seg_boundary.shape[1]):
                if seg_boundary[i, j] == 1:
                    seg_boundary[i, j] = 255
        path = "/home/yy/hust_lab/CV/github_spectral_clustering/reconstruct_project/segmentation_results/our_cut/" + image_name + "_" + str(num_cut) + ".png"
        imsave(path, seg_boundary, cmap='gray')


        row = m_img.shape[0]
        col = m_img.shape[1]
        segmentsLabel = []
        for i in range(row):
            for j in range(col):
                l = segments[i, j]
                if l not in segmentsLabel:
                    segmentsLabel.append(l)
        position = []
        for i in segmentsLabel:
            pixel_position = []
            for m in range(row):
                for n in range(col):
                    if segments[m, n] == i:
                        pixel_position.append([m, n])
            position.append(pixel_position)

        avePos = []
        for i in range(len(position)):
            cori = 0
            corj = 0
            for j in position[i]:
                [m, n] = j
                cori += m
                corj += n
            avePos.append([cori/len(position[i]), corj/len(position[i])])

        plot_position = []
        distance_label = []
        num_marker = 0
        vector_space_label = []
        spe_color_position = []
        # vector_space_label = np.zeros(len(position))
        label_segments = deepcopy(segments)
        # for i in range(len(position)):
        for i in segmentsLabel:
            plot_position.append(avePos[i])
            distance_label.append(num_marker)
            num_marker += 1
            vector_space_label.append(1)
            spe_color_position.append(avePos[i])
            # for pos in position[i]:
            #     [m, n] = pos
            #     label_segments[m, n] = 1000

        # show the numbers of SLIC
        fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.imshow(mark_boundaries(image, segments))
        # plt.imshow(mark_boundaries(image, segments, color=(0, 0, 1)))
        plt.imshow(mark_boundaries(m_img, label_segments, color=(0, 0, 1)))
        plot_position = np.asarray(plot_position)
        # plt.scatter(plot_position[:, 1],  plot_position[:, 0])
        for x, y, z in zip(plot_position[:, 1], plot_position[:, 0], distance_label):
            pl.text(x, y, str(z), color="red", fontsize=12)
            # pl.plot(plot_position[:, 1],  plot_position[:, 0], distance_label, color="red", fontsize=12)

#         plt.imshow(m_img)
# #         print("labels---------------------_", labels)
        
#         cmap = plt.cm.get_cmap("Spectral")
#         colors = cmap(a / b)

#         for l in range(len(labels)):
#             plt.contour(p_label==l, contour=1, colors=cmap(l/float(len(labels))))
#         plt.title("our method, distance=|R-R'|+|G-G'|+|B-B'|")

    matplotlib.interactive(False)
    plt.show()

