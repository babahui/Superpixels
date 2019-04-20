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
from admm_algorithms import admm, relation_density_admm

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

def large_image_seg(img_name="train/35070", sp_num=1000, sp_mode="similarity", sp_connectivity=2, num_cuts=3, admm_met='admm', thres_cons=3, n_iter=100, graph_met='lib_met', dens_coff=3):
    # read image
    base_name = os.getcwd()
    m_img = scipy.misc.imread(join(base_name, img_name))
    
    # do the superpixels method
    segments = slic(m_img, compactness=30, n_segments=sp_num)
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
        
    if graph_met == 'syn_met':
        ev1, ev2, ev3 = syn_graph_met(m_img, segments)
        
    else:
        warnings.warn('admm_met argument missing', UserWarning)
        

    # matrix = matrix.toarray()
    matplotlib.interactive(True)

            
    if admm_met == 'admm':
        sp_label = admm(n_vector=ev2, n_iter=n_iter, num_cuts=num_cuts)

    elif admm_met == 'density_admm':
        sp_label = relation_density_admm(n_vector=ev2, num_cuts=num_cuts, n_iter=n_iter, thres_cons=thres_cons)
        
    elif admm_met == 'boundary_admm':
        sp_label = bound_density_admm(n_vector=ev2, num_cuts=num_cuts, n_iter=n_iter, dens_coff=dens_coff)

    else:
        warnings.warn('admm_met argument missing', UserWarning)

        
    p_label, labels = pixels_label(m_img, segments, sp_label)
#     print(p_label.shape)
#     print(len(labels))
    
    #image_plot(m_img, p_label, labels)
    return p_label, labels, ev1, ev2, ev3, sp_label

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

def syn_graph_met(m_img, segments):
    image = m_img

    # init graph by first method, by color distance metric between superpixels.
    row = image.shape[0]
    col = image.shape[1]
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

    # fully connected
    sigma = 255.0
    length = len(position)
    graph = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            # diff = 0.299 * abs(red_average[i]-red_average[j]) + 0.587 * abs(green_average[i]-green_average[j]) + 0.114 * abs(blue_average[i]-blue_average[j])
#             diff = abs(red_average[i]-red_average[j]) + abs(green_average[i]-green_average[j]) + abs(blue_average[i]-blue_average[j])
            diff = abs(average[i] - average[j])
            graph[i, j] = diff
            # diff = abs(red_average[i]-red_average[j]) + abs(green_average[i]-green_average[j]) + abs(blue_average[i]-blue_average[j])
            # graph[i, j] = math.e ** (-(diff ** 2) / sigma)

    # # distance of superpixels not neighbor = const + Gasssian var
    # # inf_matrix = deepcopy(matrix)
    # # for i in range(matrix.shape[0]):
    # #     for j in range(matrix.shape[1]):
    # #         if matrix[i, j] == 0:
    # #             inf_matrix[i, j] = 1 + np.random.rand()

    # matplotlib.interactive(True)

    # generate graph matrix
    # g = gg.rag_mean_color(m_img, segments, mode=sp_mode, connectivity=sp_connectivity)
    # w = nx.to_scipy_sparse_matrix(g, format='csc')
    # entries = w.sum(axis=0)
    # d = sparse.dia_matrix((entries, 0), shape=w.shape).tocsc()
    # m = w.shape[0]
    # d2 = d.copy()
    # d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)
    # matrix = d2 * (d - w) * d2

    # graph = w.toarray()
    # const = 100
    # for i in range(graph.shape[0]):
    #     for j in range(graph.shape[1]):
    #         if graph[i, j] == 0:
    #             graph[i, j] = const
    #         else:
    #             diff = abs(red_average[i]-red_average[j]) + abs(green_average[i]-green_average[j]) + abs(blue_average[i]-blue_average[j])
    #             print(diff)
    #             graph[i, j] = diff

    # matrix eigen-decomposition, scipy.sparse.linalg
    vals, vectors = np.linalg.eigh(graph)
    vals, vectors = np.real(vals), np.real(vectors)
    index1, index2, index3 = np.argsort(vals)[0], np.argsort(vals)[1], np.argsort(vals)[2]
    # index1, index2, index3 = np.argsort(vals)[::-1][0], np.argsort(vals)[::-1][1], np.argsort(vals)[::-1][2]
    ev1, ev2, ev3 = vectors[:, index1], vectors[:, index2], vectors[:, index3]
    
#     sp_label = admm(n_vector=ev3, num_cuts=num_cuts)

    # # eigenvalues
    # fig = plt.figure()
    # plt.plot(vals)

#     matplotlib.interactive(True)

    # label1 = [82, 83, 81, 80]
    # label2 = [95, 87, 110, 106, 84, 135]
    # label3 = [144, 164, 178, 149, 168, 151, 169, 155, 172, 145, 156, 173, 148, 165, 182, 138, 158, 174, 139, 162, 179, 142, 160, 175, 140, 159, 176, 146, 163, 181, 130, 143, 134, 136, 153, 167, 170, 157, 177]
    # print("-----------------------sp label-------------------------", sp_label)
    # sp_label[label1] = 40
    # sp_label[label2] = 20
    # sp_label[label3] = 30

#     fig = plt.figure()
#     ax1 = Axes3D(fig, elev=-150, azim=110)
#     ax1.scatter(ev1, ev2, ev3, c=sp_label)
#     ax1.set_xlabel("x")
#     ax1.set_ylabel("y")
#     ax1.set_zlabel("z")
#     # plt.title("distance=|R-R'|+|G-G'|+|B-B'|")
#     plt.show()

    return ev1, ev2, ev3



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

