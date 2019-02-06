import sparse as sp
import numpy as np
import matplotlib.pyplot as plt

def main():
    '''
    La fonction main test les encodages de Yale sur les images
    :return: Les tests nécessaires.
    '''
   
    # Question 2

    mnist_dataset = np.memmap('train-images-idx3-ubyte', offset=16, shape=(60000, 28, 28))
    first_image = mnist_dataset[0].tolist()

    # Encodage de la première image dans une matrice éparse
    # création de fromiter: itérable de triplets(i, j, v)
    # (convertir le bitmap en une liste de coordonnées)
    fromiter = encode2D(first_image)
    # shape
    shape = shape2D(first_image)
    # matrice éparse
    sparse_first_image = sp.SparseMatrix(fromiter, shape)

    # Décodage à l'aide de todense()
    dense_first_image = sparse_first_image.todense()

    # Comparaison pixel par pixel
    boolean = first_image == dense_first_image
    print('Comparaison de la première image du MNIST avec la méthode to dense()')
    print("Affiche vrai si les deux images sont identiques:\n",boolean) # affiche vrai si les images sont identiques

    # Question 4

    # Comparaison de toutes les images
    # Les images sont toutes mises dans une même liste. 
    # on forme ainsi un seul tenseur 3D compact
    tenseur = [i.tolist() for i in mnist_dataset]
    fromiter = encode3D(tenseur)
    shape = shape3D(tenseur)
    # version dense encodée avec Yale
    sTenseur = sp.SparseTensor(fromiter,shape)
    dense_tenseur = sTenseur.todense()
    # On compare toutes les images pixel par pixel avec np.array_equal() et des boucles imbriquées
    boolean = True
    for i in range(len(tenseur)):
        for j in range(len(tenseur[0])):
            if not np.array_equal(tenseur[i][j],dense_tenseur[i][j]):
                boolean= False
                break
 
    print(' Les images sont mises dans un seul tenseur. On compare un à un les éléments avec des boucles imbriquées.\n Retourne vrai si tous les éléments sont identiques:\n',boolean)
    
    # Question 5

    # on inverse x et z (d1 et d2)
    inverse_tenseur = np.moveaxis(tenseur,0, 2)
    fromiter = encode3D(inverse_tenseur)
    shape = shape3D(inverse_tenseur)
    s_inverse_tenseur = sp.SparseTensor(fromiter,shape)

    # nouveau length de l'encodage:
    print('\n\n On inverse les axes x et z du tenseur initial pour réduire le coût en mémoire de rowptr.')
    print('length de rowptr du tenseur initial:',len(sTenseur.rowptr),"\nlength du tenseur après l'inversion des axes x et z:",len(s_inverse_tenseur.rowptr))
    print('\nOn a un gain en mémoire de len(rowptr du tenseur initial)-len(rowptr tenseur efficace) =',len(sTenseur.rowptr)-len(s_inverse_tenseur.rowptr))
    print('Ce qui représente une économie pour rowptr de:',(1-len(s_inverse_tenseur.rowptr)/len(sTenseur.rowptr))*100,'%')
    print('\nlength de colind et data du tenseur initial:',len(sTenseur.colind),"\nlength de colind et data du tenseur inversé:",len(s_inverse_tenseur.colind))
    


'''
Fonctions nécessaires à l'encodage de fromiter pour une matrice 2D
'''

def encode2D(matrix):
    '''
    :param matrix: bitmap 2 dimensions
    :return: une liste de triple de coordonnees (i,j,value)
    '''
    shape = shape2D(matrix)  # verifier pourquoi c'est impossible d'utiliser [i,j]
    fromiter = [(i, j, matrix[i][j]) for i in range(shape[0]) for j in range(shape[1]) if matrix[i][j] != 0]
    return fromiter


def shape2D(matrix):
    '''
    :param matrix: bitmap 2 dimensions
    :return: un tuple de dimensions (n,m) de la matrice
    '''
    return (len(matrix), len(matrix[0]))

'''
Fonctions nécessaires à l'encodage de fromiter pour une matrice 3D
'''

def encode3D(matrix):
    '''
    :param matrix: une matrice 3D
    :return: itérable de coordonnées des valeurs non-nulles
    '''
    shape = shape3D(matrix)
    fromiter = [(i, j, k, matrix[i][j][k]) for i in range(shape[0]) for j in range(shape[1]) for k in range(shape[2]) if
                matrix[i][j][k] != 0]
    return fromiter


def shape3D(matrix):
    '''
    :param matrix: bitmap 3 dimensions
    :return: un triplet de dimensions (x, y, z)
    '''
    return (len(matrix), len(matrix[0]), len(matrix[0][0]))

main()