from os import path

import numpy as np
import csv


def initialize_vector(pers_vector):
    """
    Normalise un vecteur
    :param pers_vector: Le vecteur de personnalisation
    :return: Le vecteur de personnalisation normalisé
    """
    return np.matrix(pers_vector / pers_vector.sum()).T


def itemRank(A: np.matrix , alpha: float, v: np.array, m: bool): #−> np.array
    """
    Calcul du vecteur de score récursivement ou par inversion matricielle (en fonction de m)
    :param A: La matrice d'adjacence
    :param alpha: Le paramètre de téléportation
    :param v: Le vecteur de personnalisation d’un utilisateur
    :param m: Une variable booléenne contenant true si le score doit être obtenu par récurrence et false s’il doit être
    obtenu par inversion matricielle.
    :return: Un vecteur contenant les scores d’importance des noeuds ordonnés dans le même ordre que la matrice
    d’adjacence
    """
    PTransposed = get_probability_transition_matrix(A).T
    di = initialize_vector(v)
    if m:
        res = item_rank_recursively(PTransposed, alpha, di, di)
    else:
        res = item_rank_inversion_mat(PTransposed, alpha, di)
    return np.squeeze(np.asarray(res))


def item_rank_recursively(PTransposed: np.matrix , alpha: float, xi,  v: np.matrix):
    """
    Calcul du vecteur de score récursivement
    :param PTransposed: La transposée de la matrice de probabilités de transition
    :param alpha: Le paramètre de téléportation
    :param xi: Le vecteur de page rank actuel
    :param v: Le vecteur de personnalisation normalisé
    :return: Le vecteur de page rank
    """
    newVect = alpha * PTransposed * xi + (1 - alpha) * v
    if np.sum(np.abs(xi - newVect)) <= PRECISION:
        return newVect
    return item_rank_recursively(PTransposed, alpha, newVect, v)


def item_rank_inversion_mat(PTransposed: np.matrix , alpha: float, v: np.array):
    """
    Calcul du vecteur de score par inversion matricielle
    :param PTransposed: La transposée de la matrice de probabilités de transition
    :param alpha: Le paramètre de téléportation
    :param v: Le vecteur de personnalisation normalisé
    :return: Le vecteur de page rank
    """
    I = np.identity(PTransposed.__len__())
    res = (1 - alpha) * (np.linalg.inv(I - alpha*PTransposed)) * v
    return res


def get_probability_transition_matrix(A: np.matrix):
    """
    Calcule la matrice de probabilité de transition à partir de la matrice d'adjacence
    :param A: La matrice de base
    :return: La matrice de probabilités de transition
    """
    res = []
    for i in range(A.__len__()):
        res.append([0] * A.__len__())
    do = A.sum(axis=1)
    for i in range(A.__len__()):
        res[i] = (A[i]/float(do.item(i))).A1
    return np.matrix(res)


def read_csv(filename):
    """
    Lit un fichier csv pour retourner son contenu dans un tableau
    :param filename: Le nom du fichier devant être lu
    :return: Le contenu du fichier dans un tableau
    """
    res = []
    file = path.join(path.join(path.join(path.dirname(__file__), '..'), 'Code'), filename)
    with open(file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            res.append([float(elem) for elem in row])
    return res


def main():
    """
    Fonction principale dans laquelle les fonctions sont appelée
    """
    personalisation_vector = np.array(read_csv(NOM_DU_FICHIER_CSV_VECTEUR_PERSO)[0])
    matrix = np.matrix(read_csv(NOM_DU_FICHIER_CSV_MATRICE_ADJACENCE))

    recursively = itemRank(matrix, ALPHA, personalisation_vector, True)
    print("item rank par récursivité\n", recursively)

    inversion = itemRank(matrix, ALPHA, personalisation_vector, False)
    print("item rank par inversion matricielle\n", inversion)


if __name__ == '__main__':
    """
    La valeur d'alpha, la valeur de la précision du vecteur de score calculé par récurrence 
    et les noms des fichiers csv peuvent être changés ici.
    """
    ALPHA = 0.15
    NOM_DU_FICHIER_CSV_MATRICE_ADJACENCE = "matrixBase.csv"
    NOM_DU_FICHIER_CSV_VECTEUR_PERSO = "Personnalisation_Student30.csv"
    PRECISION = 0.00001

    main()
