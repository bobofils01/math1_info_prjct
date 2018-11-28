import numpy as np


def initialize_vector(n):
    return np.zeros((n), dtype=np.int16) #[0 for i in range(n)]


def calculation_vector():
    a = initialize_vector(4)
    a[3] = 4
    print(a)


def itemRank (A: np.matrix , alpha: float, v: np.array, m: bool): #−> np.array

    return []


if __name__ == '__main__':
    print("Item Rank", initialize_vector(3))
    calculation_vector()

