import math

import numpy as np

def normal_list(xs, mu=0, sigma=1):
    vals = np.array([])
    for x in xs:
        vals = np.append(vals, normal_single(x, mu=mu, sigma=sigma))

    return vals


def normal_single(x, mu=0, sigma=1):
    coef = 1 / math.sqrt(2 * math.pi * (sigma ** 2))

    numerator = - ((x - mu) ** 2)
    denominator = 2 * (sigma ** 2)
    return coef * math.exp(numerator / denominator)