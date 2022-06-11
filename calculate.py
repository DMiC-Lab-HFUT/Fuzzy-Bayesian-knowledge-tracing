import numpy as np

def cal_membership(y, miu, sigma):

    if sigma < 0.0000001:
        sigma += 0.0000001
    former = 1 / (np.sqrt(2 * np.pi) * sigma)

    inx = 0.5 * (((y - miu) / (sigma)) ** 2)
    if inx >= 0:
        member = former * np.exp(-inx)
    else:
        member = former / np.exp(inx)
    return member



def cal_distance(x, y):
    return abs(x - y)
