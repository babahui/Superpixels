"""
===============
admm algorithms
===============

A partition method with ADMM method, code by pure python.
"""

# print(__doc__)

import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import math

def admm(n_vector, num_cuts=2, n_iter=1000, merge=False):
    """  Assign labels to input vector.

    Minimize cost funtion: argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a,
    while x is the input vector, a is cut_number.

    Parameters
    ----------
    num_cuts : int
        Number of iteration will cut on this vector/eigenvector.
    n_iter : int
        itertion times of sub-problems.
    n_vector : ndarray
        Input vector, as x in the cost function.

    Returns
    -------
    out : ndarray
        The new labeled array.

    """

    # sort the input vector and get the sorted index
    sort_index = np.argsort(n_vector)
    sort_vector = [n_vector[i] for i in sort_index]
    sort_vector = np.asarray(sort_vector)
    length = len(sort_vector)

    # initilize variables of cost function
    A = np.zeros((length-1, length))
    for i in range(length-1):
        A[i][i] = -1
        A[i][i+1] = 1
    I = np.identity(length)
    iteration = 0
    x = deepcopy(sort_vector)
    x1 = deepcopy(sort_vector)
    v = np.dot(A, x1)
    w = np.dot(A, x1)
    r = 0.01
    n = 0.99 # n belong to [0.95, 0.99]

    ''' solve cost function by three sub-equations,
    x' = inv(I + 1/r * A.T * A) * (x1 + 1/r * A.T * (v - w))
    v' = A * x + w, if |z|_0 <= a,
      or sort(A*x+w)[:a] , if |z|_0 > a
    w' = w + A * x - v
    '''
    # left part of x out of while block to save computation time.
    left = np.linalg.inv(np.dot(1, I) + np.dot(1/r, np.dot(A.T, A)))
    while iteration < n_iter:
        right = np.dot(1, x1) + np.dot(1/r, np.dot(A.T, (v - w)))
        x = np.dot(left, right)
        z = np.dot(A, x) + w
        if np.linalg.norm(z, 0) <= num_cuts:
            v = deepcopy(z)
            # print("z<a")
        else:
            z_abs = abs(z)
            z_sort = np.argsort(z_abs)[::-1]
            v = np.zeros(length-1)
            for i in range(num_cuts):
                index = z_sort[i]
                v[index] = z[index]
        w = w + np.dot(A, x) - v

        # It seems reducing r in each iteration will converage fast (?)
        # r = r * n
        iteration += 1

    # assign labels by final gaps
    gap = np.dot(A, x)
    # print("___________gap_____________:", gap)
    k = num_cuts
    big_k_gap = np.sort(np.argsort(gap)[::-1][:k])
#     print(big_k_gap)
    assign_label = np.zeros(length)
    label = 1
    for i in range(len(big_k_gap)-1):
        begin_indice = big_k_gap[i] + 1
        end_indice = big_k_gap[i+1] + 1
#         print("begin and end", begin_indice, end_indice)
        assign_label[begin_indice:end_indice] = label
        label += 1
    assign_label[big_k_gap[-1]:] = label

    label_vector = np.zeros(length)
    for i in range(length):
        label_vector[sort_index[i]] = assign_label[i]
    # labelVec = np.zeros(len(unSortedLabelVec))
    # for i in range(len(unSortedLabelVec)):
    #     labelVec[sortIndex[i]] = unSortedLabelVec[i]
    
    # sort_vector plot
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.scatter(n_vector, np.zeros(len(sort_vector)), c=label_vector)
#     plt.show()
    
    if merge:
        import time
        start_time = time.time()
        label_vector = class_merge(n_vector, label_vector, merge)    
        print('merge time: ', time.time()-start_time)
    return label_vector


def class_merge(n_vector, label_vector, merge):
    # if min ,then merge
    count = 0

    while count < merge:
        labels = []
        for label in label_vector:
            if label not in labels:
                labels.append(label)
        
        num = []
        pos = []
        for label in labels:
            lab_pos = []
            for ith in range(len(n_vector)):
                if label_vector[ith] == label:
                    lab_pos.append(ith)
                    
            pos.append(lab_pos)
            num.append(len(lab_pos))
            
        # merge rules:
        min_num = 10000
        for i in range(0, len(pos)-1):
            for j in range(i+1, len(pos)):
                if num[i] + num[j] < min_num:
                    c_1 = i
                    c_2 = j
                    min_num = num[i] + num[j]
                    
        # update label_vector
        for p in pos[c_2]:
            label_vector[p] = labels[c_1]
            
        count += 1
        
    return label_vector

# relation order between, e-graph
def relation_density_admm(n_vector, num_cuts=2, n_iter=100, thres_cons=3):

    # sort the input vector and get the sorted index
    sort_index = np.argsort(n_vector)
    sort_vector = [n_vector[i] for i in sort_index]
    sort_vector = np.asarray(sort_vector)
    length = len(sort_vector)

    # initilize variables of cost function
    A = np.zeros((length-1, length))
    for i in range(length-1):
        A[i][i] = -1
        A[i][i+1] = 1
    I = np.identity(length)
    iteration = 0
    x = deepcopy(sort_vector)
    x1 = deepcopy(sort_vector)
    v = np.dot(A, x1)
    w = np.dot(A, x1)
    r = 0.01
    n = 0.99 # n belong to [0.95, 0.99]

    ''' solve cost function by three sub-equations,
    x' = inv(I + 1/r * A.T * A) * (x1 + 1/r * A.T * (v - w))
    v' = A * x + w, if |z|_0 <= a,
      or sort(A*x+w)[:a] , if |z|_0 > a
    w' = w + A * x - v
    '''
    # left part of x out of while block to save computation time.
#     C = relation_density(x1, thres_cons=thres_cons)

#     left = np.linalg.inv(np.dot(1, I) + np.dot(1/r, np.dot(A.T, A)))
    while iteration < n_iter:
        C = relation_density(x1, thres_cons=thres_cons)
        
        left = np.linalg.inv(np.dot(C, I) + np.dot(1/r, np.dot(A.T, A)))
        right = np.dot(C, x1) + np.dot(1/r, np.dot(A.T, (v - w)))
        x = np.dot(left, right)
        z = np.dot(A, x) + w
        if np.linalg.norm(z, 0) <= num_cuts:
            v = deepcopy(z)
            # print("z<a")
        else:
            z_abs = abs(z)
            z_sort = np.argsort(z_abs)[::-1]
            v = np.zeros(length-1)
            for i in range(num_cuts):
                index = z_sort[i]
                v[index] = z[index]
        w = w + np.dot(A, x) - v

        # It seems reducing r in each iteration will converage fast (?)
        # r = r * n
        iteration += 1

    # assign labels by final gaps
    gap = np.dot(A, x)
    # print("___________gap_____________:", gap)
    k = num_cuts
    big_k_gap = np.sort(np.argsort(gap)[::-1][:k])
#     print(big_k_gap)
    assign_label = np.zeros(length)
    label = 1
    for i in range(len(big_k_gap)-1):
        begin_indice = big_k_gap[i] + 1
        end_indice = big_k_gap[i+1] + 1
#         print("begin and end", begin_indice, end_indice)
        assign_label[begin_indice:end_indice] = label
        label += 1
    assign_label[big_k_gap[-1]:] = label

    label_vector = np.zeros(length)
    for i in range(length):
        label_vector[sort_index[i]] = assign_label[i]
    # labelVec = np.zeros(len(unSortedLabelVec))
    # for i in range(len(unSortedLabelVec)):
    #     labelVec[sortIndex[i]] = unSortedLabelVec[i]
    return label_vector

def relation_density(vector, thres=None, thres_cons=3):
    if thres == None:
        thres = (vector[-1] - vector[0]) / len(vector)
        thres = thres * thres_cons
    
    # create distance matrix
    length = len(vector)
    dis_mat = np.zeros((length, length))

    for i in range(length):
        for j in range(length):
            dis_mat[i, j] = abs(vector[i] - vector[j])
            
    # convert to epsional matrix
    # get the relation count number, generate the count number vector
    count_mat = np.zeros((length, length))
    for i in range(length):
        count_num = 0
        for j in range(length):
            if dis_mat[i, j] <= thres:
                count = 1
            else:
                count = 0
            count_num += count
        count_mat[i, i] = count_num
#     return count_mat
    return np.dot(count_mat.T, count_mat)



# def var_admm(n_vector = n_vector):
#     iter_count = 0
#     while iter_count < 1:
#         label_vector = admm(n_vector, num_cuts=2, n_iter=1000)
        
#         new_label_vector = adjust_by_variance(label_vector, n_vector)
# #         n_vector = new_vector

#         iter_count += 1
      
#     return new_label_vector

# def adjust_by_variance(label_vector, n_vector):
    

    
def var_admm(n_vector, num_cuts=2, n_iter=1000):
    
    # sort the input vector and get the sorted index
    sort_index = np.argsort(n_vector)
    sort_vector = [n_vector[i] for i in sort_index]
    sort_vector = np.asarray(sort_vector)
    length = len(sort_vector)

    # initilize variables of cost function
    A = np.zeros((length-1, length))
    for i in range(length-1):
        A[i][i] = -1
        A[i][i+1] = 1
    I = np.identity(length)
    iteration = 0
    x = deepcopy(sort_vector)
    x1 = deepcopy(sort_vector)
    v = np.dot(A, x1)
    w = np.dot(A, x1)
    r = 0.01
    n = 0.99 # n belong to [0.95, 0.99]

    ''' solve cost function by three sub-equations,
    x' = inv(I + 1/r * A.T * A) * (x1 + 1/r * A.T * (v - w))
    v' = A * x + w, if |z|_0 <= a,
      or sort(A*x+w)[:a] , if |z|_0 > a
    w' = w + A * x - v
    '''
    # left part of x out of while block to save computation time.
    left = np.linalg.inv(np.dot(1, I) + np.dot(1/r, np.dot(A.T, A)))
    while iteration < n_iter:
        right = np.dot(1, x1) + np.dot(1/r, np.dot(A.T, (v - w)))
        x = np.dot(left, right)
        z = np.dot(A, x) + w
        if np.linalg.norm(z, 0) <= num_cuts:
            v = deepcopy(z)
            # print("z<a")
        else:
            z_abs = abs(z)
            z_sort = np.argsort(z_abs)[::-1]
            v = np.zeros(length-1)
            for i in range(num_cuts):
                index = z_sort[i]
                v[index] = z[index]
        w = w + np.dot(A, x) - v

        # It seems reducing r in each iteration will converage fast (?)
        # r = r * n
        iteration += 1

    # assign labels by final gaps
    indice = []
    
    gap = np.dot(A, x)
    # print("___________gap_____________:", gap)
    k = num_cuts
    big_k_gap = np.sort(np.argsort(gap)[::-1][:k])
#     print(big_k_gap)
    assign_label = np.zeros(length)
    label = 1
    for i in range(len(big_k_gap)-1):
        begin_indice = big_k_gap[i] + 1
        end_indice = big_k_gap[i+1] + 1
#         print("begin and end", begin_indice, end_indice)
        assign_label[begin_indice:end_indice] = label
        label += 1
        # 拿到索引
        indice.append(begin_indice)
        indice.append(end_indice)
        
    assign_label[big_k_gap[-1]:] = label

    # 矫正
    assign_label = adjust_by_variance(sort_vector, assign_label, indice)
    
    
    label_vector = np.zeros(length)
    for i in range(length):
        label_vector[sort_index[i]] = assign_label[i]
    # labelVec = np.zeros(len(unSortedLabelVec))
    # for i in range(len(unSortedLabelVec)):
    #     labelVec[sortIndex[i]] = unSortedLabelVec[i]
    return label_vector


def adjust_by_variance(sort_vector, assign_label, indice): 
    # 不改变簇数
    print('----------------indice---------------------', indice)

    for i in indice:
        best_var = variance(assign_label, sort_vector)
        count = 0
        num = 0
        while count < 100 and i < len(sort_vector)-1:
            count += 1

            if assign_label[i] != assign_label[i+1]:
                test_label = assign_label.copy()
                test_label[i+1] = test_label[i]
                
                if variance(test_label, sort_vector) < best_var:
                    best_var = variance(test_label, sort_vector)
                    # change label
                    assign_label[i+1] = assign_label[i]
                    num += 1

                else:
                    break
            i = i + 1
        print('-------num----------', num)    
    return assign_label
                 

def variance(label, vector):
    uniq_labels = []
    for l in label:
        if l not in uniq_labels:
            uniq_labels.append(l)
            
    var_vector = []        
    for i in uniq_labels:
        var_vector.append([vector[k] for k in range(len(label)) if label[k] == i])

    var = 0
    for v in var_vector:
        var += np.var(v)
    
    return var

#     return sum(np.var(var_vector, axis=0))
    


if __name__ == "__main__":
    input_vector = np.random.rand(1000)
    # label_vector = admm(input_vector=input_vector)
    label_vector = admm(n_vector=input_vector)
    print(label_vector)
    plt.scatter(input_vector, np.ones(len(input_vector)), c=label_vector)
    plt.show()
