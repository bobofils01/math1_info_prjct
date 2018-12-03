import numpy as np

matrice = np.matrix([
   # 1  2  3  4  5  6  7  8  9  10
    [0, 1, 1, 0, 1, 0, 1, 1, 1, 1], #1
    [1, 0, 0, 1, 1, 0, 1, 1, 0, 0], #2
    [1, 0, 0, 0, 1, 1, 1, 1, 1, 0], #3
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0], #4
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 0], #5
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0], #6
    [1, 1, 1, 0, 1, 1, 0, 0, 1, 1], #7
    [1, 1, 1, 0, 1, 1, 0, 0, 1, 0], #8
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 0], #9
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]  #10
                                        ])

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

