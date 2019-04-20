"""
===============
admm algorithms
===============

A partition method with ADMM method, code by pure python.
"""

print(__doc__)

import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import math

def admm(n_vector, num_cuts=2, n_iter=1000):
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
    return label_vector

# relation order between, e-graph
def relation_density_admm(n_vector, num_cuts=2, n_iter=100, thres_cons=3):
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

    
# create a density matrix representation 
def bound_density_admm(n_vector, num_cuts=2, n_iter=10000, dens_coff=3):
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
    Dens_mat = bound_density(dens_coff=dens_coff, dim=len(x1))
    left = np.linalg.inv(np.dot(Dens_mat, I) + np.dot(1/r, np.dot(A.T, A)))
    while iteration < n_iter:
        right = np.dot(Dens_mat, x1) + np.dot(1/r, np.dot(A.T, (v - w)))
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
    print(big_k_gap)
    assign_label = np.zeros(length)
    label = 1
    for i in range(len(big_k_gap)-1):
        begin_indice = big_k_gap[i] + 1
        end_indice = big_k_gap[i+1] + 1
        print("begin and end", begin_indice, end_indice)
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

def bound_density(dens_coff=2, dim=None):
    if dim == None:
        print("dim not found")
    # init zero density matrix
    dens_mat = np.zeros((dim, dim))
    
    padding_vec = []
    for i in range(dens_coff):
        padding_vec.append(1 / math.pow(2, (dens_coff - i)))
    padding_vec.append(1)
    for i in range(dens_coff):
        padding_vec.append(-1 / math.pow(2, (dens_coff - i)))
        
    for i in range(dim):
        if i - dens_coff < 0:
            if i == 0:
                vec = 0
            else:
                vec = []
                for j in range(i):
                    vec.append(1 / math.pow(2, i-j))
                vec.append(1)
                for j in reversed(range(i)):                
                    vec.append(1 / math.pow(2, i-j))
            dens_mat[i, :2*i+1] = vec
            
        if 0 <= i-dens_coff < i+dens_coff+1 < dim:
            dens_mat[i, i-dens_coff:i+dens_coff+1] = padding_vec
            
        if i + dens_coff + 1 >= dim:
            if i == dim - 1:
                dens_mat[i, dim-1] = 0
            else:
                vec = []
                for j in range(dim-1-i):
                    vec.append(1 / math.pow(2, dim-1-i-j))
                vec.append(1)
                for j in reversed(range(dim-1-i)):
                    vec.append(1 / math.pow(2, dim-1-i-j))
                dens_mat[i, 2*i-dim+1:] = vec 
    return np.dot(dens_mat.T, dens_mat)            
                    
def admm_without_sorting(n_vector, num_cuts=2, n_iter=10000):
    """  Assign labels to input vector.

    !!! this version didn't sort the input vector. different between admm algorithms below.

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

    length = len(n_vector)

    # initilize variables of cost function
    A = np.zeros((length-1, length))
    for i in range(length-1):
        A[i][i] = -1
        A[i][i+1] = 1
    I = np.identity(length)
    iteration = 0
    x = deepcopy(n_vector)
    x1 = deepcopy(n_vector)
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
    print(big_k_gap)
    assign_label = np.zeros(length)
    label = 1
    for i in range(len(big_k_gap)-1):
        begin_indice = big_k_gap[i] + 1
        end_indice = big_k_gap[i+1] + 1
        print("begin and end", begin_indice, end_indice)
        assign_label[begin_indice:end_indice] = label
        label += 1
    assign_label[big_k_gap[-1]:] = label

    return assign_label

def admm_vector_iter(n_vector, num_cuts=2, n_iter=10000):
    """  Assign labels to input vector.

    Minimize cost funtion: argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a,
    while x is the input vector, a is cut_number.
    每次迭代后把新的向量作为输入，直到收敛。

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

        x1 = deepcopy(x)

        iteration += 1

    # assign labels by final gaps
    gap = np.dot(A, x)
    # print("___________gap_____________:", gap)
    k = num_cuts
    big_k_gap = np.sort(np.argsort(gap)[::-1][:k])
    print(big_k_gap)
    assign_label = np.zeros(length)
    label = 1
    for i in range(len(big_k_gap)-1):
        begin_indice = big_k_gap[i] + 1
        end_indice = big_k_gap[i+1] + 1
        print("begin and end", begin_indice, end_indice)
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

def mutiple_vector_admm(eigvectors, iter_count, iteration_mode='iter_all', label=None, num_cuts=2, n_iter=10000):
    """  Assign labels to input vector.

    Minimize cost funtion: argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a,
    while x is the input vector, a is cut_number.
    全局的多维特征向量作为输入，每次迭代后更新坐标作为新的输入。返回点在特征空间中的变化。

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


   solve cost function by three sub-equations,
    x' = inv(I + 1/r * A.T * A) * (x1 + 1/r * A.T * (v - w))
    v' = A * x + w, if |z|_0 <= a,
      or sort(A*x+w)[:a] , if |z|_0 > a
    w' = w + A * x - v
   """

    cor = []

    for eig_count in range(eigvectors.shape[1]):
        n_vector = eigvectors[:, eig_count]
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
        x1 = deepcopy(sort_vector)
        v = np.dot(A, x1)
        w = np.dot(A, x1)
        r = 0.01
        n = 0.99 # n belong to [0.95, 0.99]

        eig_cor = []
        # left part of x out of while block to save computation time.
        left = np.linalg.inv(np.dot(1, I) + np.dot(1/r, np.dot(A.T, A)))
        while iteration < n_iter:
            for i_num in iter_count:
                if iteration == i_num:
                    eig_cor.append(x)
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
            if iteration_mode == "iter_all":
                x1 = deepcopy(x)
            elif iteration_mode == "iter_none":
                pass
            else:
                print("set iteration mode")
            iteration += 1
        cor.append(eig_cor)

    cor = np.asarray(cor)
    return cor

    # # assign labels by final gaps
    # gap = np.dot(A, x)
    # # print("___________gap_____________:", gap)
    # k = num_cuts
    # big_k_gap = np.sort(np.argsort(gap)[::-1][:k])
    # print(big_k_gap)
    # assign_label = np.zeros(length)
    # label = 1
    # for i in range(len(big_k_gap)-1):
    #     begin_indice = big_k_gap[i] + 1
    #     end_indice = big_k_gap[i+1] + 1
    #     print("begin and end", begin_indice, end_indice)
    #     assign_label[begin_indice:end_indice] = label
    #     label += 1
    # assign_label[big_k_gap[-1]:] = label

    # label_vector = np.zeros(length)
    # for i in range(length):
    #     label_vector[sort_index[i]] = assign_label[i]
    # # labelVec = np.zeros(len(unSortedLabelVec))
    # # for i in range(len(unSortedLabelVec)):
    # #     labelVec[sortIndex[i]] = unSortedLabelVec[i]
    # return label_vector

if __name__ == "__main__":
    input_vector = np.random.rand(1000)
    # label_vector = admm(input_vector=input_vector)
    label_vector = admm(n_vector=input_vector)
    print(label_vector)
    plt.scatter(input_vector, np.ones(len(input_vector)), c=label_vector)
    plt.show()
