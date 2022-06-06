from collections import defaultdict
from random import uniform
from math import sqrt, exp
import calculate as cal


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2
    
    Returns a new point which is the center of all the points.
    """
    # dimensions = len(points)

    new_center = []
    # dim_sum = 0
    points = [each[0] for each in points]
    # for dimension in range(dimensions):
        # dim_sum = 0  # dimension sum
    dim_sum = 0

    for point in points:
        # if point > 0:
        dim_sum += point
    new_center.append(dim_sum / float(len(points)))
    return new_center


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.

    Compute the center for each of the assigned groups.

    Return `k` centers where `k` is the number of unique assignments.
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
        
    for points in new_means.values():
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point. 
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []
    for point in data_points:
        shortest = float('inf')  # positive infinity
        shortest_index = 0
        centers.sort()
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    # print('centers is ')
    # print(centers)
    # print('assignments is ')
    # print(assignments)
    return centers, assignments


def distance(a, b):
    """
    """
    dimensions = len(a)
    
    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)


def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]
            
            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)
    centers.sort()
    return centers

##求解每个聚类的均值和标准差
def get_sigma1(dataset, centers, assignments):
    sigma1 = []
    sum_dataset = 0

    for i in range(len(centers)):
        # 求出类别assignments分别等于0,1,2,3的元素下标
        index = [j for j, x in enumerate(assignments) if x == i]
        # 求dataset对应每个类别index的标准差
        for index_i in index:
            if dataset[index_i][0] > 0:
                sum_dataset += (dataset[index_i][0] - centers[i][0]) ** 2
        sigma1.append(sqrt(sum_dataset / len(index)))
    return sigma1

#求隶属度
def get_membership(de_dataset, centers, miu1, sigma1):
    membership = []
    dict = {}
    for j in range(len(de_dataset)):
        M = []
        for q in range(len(centers)):
            m = cal.cal_membership(de_dataset[j], miu1[q][0], sigma1[q])
            M.append(m)
            # if sigma1[q] != 0:
            #     m = cal.cal_membership(de_dataset[j], miu1[q][0], sigma1[q])
            # else:
            #     #判断de_dataset[j]与miu[q][0]的距离
            #     #判断距离大小
            #     if cal.cal_distance(de_dataset[j],  miu1[q][0]) ==\
            #         max([cal.cal_distance(de_dataset[j], miu1[p][0]) for p in range(len(miu1))]):
            #         m = 1
            #     else:
            #         m = 0
            # if m > 0:
            #     M.append(m)
            # else:
            #     m = 0
            #     M.append(m)
        dict[de_dataset[j]] = [each / sum(M) for each in M]
    return dict

def k_means(dataset, k):
    k_points = generate_k(dataset, k)
    centers, assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        centers, assignments = assign_points(dataset, new_centers)
    # print('cluster finished')
    # return zip(assignments, dataset)

    ##求解每个聚类的准差
    sigma1 = get_sigma1(dataset, centers, assignments)
    miu1 = centers

    ##求解每个dataset对应每个类的隶属度
    dataset = [dataset[i][0] for i in range(len(dataset))]
    de_dataset = list(set(dataset)) #对dataset去重
    de_dataset.sort() #de_dataset排序
    dict = get_membership(de_dataset, centers, miu1, sigma1)

    return centers, sigma1, assignments, de_dataset, dict