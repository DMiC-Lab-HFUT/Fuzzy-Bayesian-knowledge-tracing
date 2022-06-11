from collections import defaultdict
from random import uniform
from math import sqrt, exp
import calculate as cal


def point_avg(points):

    new_center = []

    points = [each[0] for each in points]

    dim_sum = 0

    for point in points:
        dim_sum += point
    new_center.append(dim_sum / float(len(points)))
    return new_center


def update_centers(data_set, assignments):
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
        
    for points in new_means.values():
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
   
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

def get_sigma1(dataset, centers, assignments):
    sigma1 = []
    sum_dataset = 0

    for i in range(len(centers)):
        index = [j for j, x in enumerate(assignments) if x == i]
        for index_i in index:
            if dataset[index_i][0] > 0:
                sum_dataset += (dataset[index_i][0] - centers[i][0]) ** 2
        sigma1.append(sqrt(sum_dataset / len(index)))
    return sigma1


def get_membership(de_dataset, centers, miu1, sigma1):
    membership = []
    dict = {}
    for j in range(len(de_dataset)):
        M = []
        for q in range(len(centers)):
            m = cal.cal_membership(de_dataset[j], miu1[q][0], sigma1[q])
            M.append(m)
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
    sigma1 = get_sigma1(dataset, centers, assignments)
    miu1 = centers

    dataset = [dataset[i][0] for i in range(len(dataset))]
    de_dataset = list(set(dataset)) 
    de_dataset.sort() 
    dict = get_membership(de_dataset, centers, miu1, sigma1)

    return centers, sigma1, assignments, de_dataset, dict