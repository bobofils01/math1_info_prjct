import numpy as np


def initialize_vector(pers_vector):
    """

    :param pers_vector: Le vecteur de personnalisation
    :return: Le vecteur de personnalisation normalisé
    """
    return np.matrix(pers_vector / pers_vector.sum()).T


def itemRank(A: np.matrix , alpha: float, v: np.array, m: bool): #−> np.array
    PTransposed = get_probability_transition_matrix(A).T
    di = initialize_vector(v)
    if m:
        return item_rank_recursively(PTransposed, alpha, di, di), item_rank_inversion_mat(PTransposed, alpha, di)
    return []


def item_rank_recursively(PTransposed: np.matrix , alpha: float, xi,  v: np.matrix):
    """

    :param PTransposed: La transposée de la matrice de probabilités de transition
    :param alpha: Le paramètre de téléportation
    :param xi: Le vecteur de page rank actuel
    :param v: Le vecteur de personnalisation normalisé
    :return: Le vecteur de page rank
    """
    newVect = alpha * PTransposed * xi + (1 - alpha) * v
    if np.sum(np.abs(xi - newVect)) <= 0.00000001:
        return newVect
    return item_rank_recursively(PTransposed, alpha, newVect, v)


def item_rank_inversion_mat(PTransposed: np.matrix , alpha: float, v: np.array):
    """

    :param PTransposed: La transposée de la matrice de probabilités de transition
    :param alpha: Le paramètre de téléportation
    :param v: Le vecteur de personnalisation normalisé
    :return: Le vecteur de page rank
    """
    I = np.identity(PTransposed.__len__())
    #     (1−α)(I−αP^T)^−1*v u
    res = (1 - alpha) * (np.linalg.inv(I - alpha*PTransposed)) * v
    print("res item_rank_inversion_mat", res)
    return res


def get_probability_transition_matrix(A: np.matrix):
    """

    :param A: La matrice de base
    :return: La matrice de probabilités de transition
    """
    res = []
    for i in range(10):
        res.append([0] * 10)
    do = A.sum(axis=1)
    for i in range(A.__len__()):
        res[i] = (A[i]/float(do.item(i))).A1
    return np.matrix(res)


if __name__ == '__main__':

    ALPHA = 0.85
    matrix = np.matrix([
        # 1  2  3  4  5  6  7  8  9  10
        [0, 1, 1, 0, 1, 0, 1, 1, 1, 1],  # 1
        [1, 0, 0, 1, 1, 0, 1, 1, 0, 0],  # 2
        [1, 0, 0, 0, 1, 1, 1, 1, 1, 0],  # 3
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # 4
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0],  # 5
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 6
        [1, 1, 1, 0, 1, 0, 0, 0, 1, 1],  # 7
        [1, 1, 1, 0, 1, 1, 0, 0, 1, 0],  # 8
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 0],  # 9
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # 10
    ])

    personalisation_vector = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 1])

    #personalisation_vector = initialize_vector(personalisation_vector)
    print("personalisation_vector", personalisation_vector)
    print("resFinal", itemRank(matrix, ALPHA, personalisation_vector, True))
    #item_rank_inversion_mat(matrix, personalisation_vector)