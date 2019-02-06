import numpy as np

class SparseMatrix:
    '''
    Encodage de Yale 2D
    '''
    def __init__(self, fromiter, shape):
        '''
        :param fromiter: itérable de triplets des coordonées des valeurs non-nulles (i,j,v)
        :param shape: tuple de dimensions de la matrice
        '''
        n, m = shape # O(1)
        self.n = n # O(1)
        self.m = m # O(1)
        self.nnz = len(fromiter)  # nombre de non-zero. O(1)
        self.rowptr = [0]  # liste de taille n+1 des intervalles des colonnes. O(1)
        self.colind = []  # liste de taille nnz des indices des valeurs non-nulles. O(1)
        self.data = []  # liste de taille nnz des valeurs non-nulles. O(1)

        sfromiter = sorted(fromiter) # pour ne pas avoir à gérer les indices. O(nlogn)

        '''rowptr O(n*n*m)
        '''
        # on compte le nombre de nnz par rangée O(nnz*n)
        d = {}  # { rangee: nombre d'occurences de non-zero }. O(1) 

        for i in range(len(fromiter)): # O(nnz)

            # On ne doit pas fournir des zéros au dictionnaire
            if fromiter[i][-1] == 0:
                continue

            row = fromiter[i][0] # O(1)

            if row not in d:  # si la rangee n'est pas encore dans le dictionnaire. O(n)
                d[row] = 1  # on ajoute une occurence de la rangee. O(n)
            else:
                d[row] += 1  # on incremente le nombre d'occurences de la rangee. O(n)

        # on crée les intervalles O(n*n)
        for i in range(self.n): # O(n)

            row = self.rowptr[i]  # borne inférieure de l'intervalle. O(n)

            if i in d:  # si la rangée contient un/des non-zero. O(n)
                self.rowptr.append(row + d[i])  # on ajoute l'intervalle O(n)
            else:
                self.rowptr.append(row)  # l'intervalle est nul. O(n)

        '''colind & data O(nnz)
        '''
        for i in sfromiter: # O(nnz)

            self.colind.append(i[1]) #O(1)
            self.data.append(i[2]) # O(1)

            
    def __getitem__(self, k):
        '''
        :param k: tuple d'indices (i,j)
        :return: la valeur à la position (i,j)
        '''
        i, j = k
        # on cherche la valeur a l'indice avec une recherche dichotomique
        index = binarySearch(self.colind, j, self.rowptr[i], self.rowptr[i + 1]-1)

        if index == -1:
            return 0

        return self.data[index]


    def todense(self):
        '''
        :return: Les données dans le format d'une matrice
        '''
        x, y = self.n, self.m
        matrix = np.zeros((x, y), dtype=int) # matrice nulle

        for i in range(len(self.rowptr) - 1):

            indices = range(self.rowptr[i], self.rowptr[i + 1])

            for j in indices:

                matrix[i][self.colind[j]] = self.data[j] # on affecte les nnz aux bons endroits

        return matrix.tolist()



class SparseTensor:
    '''
    Tenseurs 3D de l'encodage de Yale
    '''
    def __init__(self, fromiter, shape):
        '''
        Tenseurs 3D de l'encodage de Yale
        '''
        n, m, z = shape
        self.n = n
        self.m = m
        self.z = z  
        self.nnz = len(fromiter)  # nombre de non-zero
        self.rowptr = [0]  # liste de taille n+1 des intervalles des colonnes
        self.colind = []  # liste de taille nnz des indices des valeurs non-nulles (2e dimension)
        self.data = []  # liste de taille nnz des valeurs non-nulles

        sfromiter = sorted(fromiter)

        '''rowptr
        '''
        d = {}

        for i in range(len(fromiter)):

            # On ne doit pas fournir des zéros au dictionnaire
            if fromiter[i][-1] == 0:
                continue

            key = (fromiter[i][0],fromiter[i][1]) # clef correspondant a la dimension la plus imbriquée

            if key not in d: # si la rangée n'est pas présente
                d[key] = 1 # on ajoute une occurance 
            else:
                d[key] += 1

        # on crée les intervalles
        indice = 0

        for i in range(self.n):

            for j in range(self.m):

                key = (i,j)

                if key in d:
                    self.rowptr.append(self.rowptr[indice]+d[key])
                else:
                    self.rowptr.append(self.rowptr[indice])

                indice += 1

        '''colind & data
        '''
        for i in fromiter:

            self.colind.append(i[2])
            self.data.append(i[3])


    def todense(self):
        '''
        :return: Les données dans le format traditionnel d'une matrice. matrice n*m*z. 
        '''
        x, y, z = self.n, self.m, self.z # dimensions
        matrix = np.zeros((x, y, z), dtype=int) # matrice x*y*z de 0

        for a in range(len(self.rowptr) - 1): # on traverse les intervalles de rowptr.

            indices = range(self.rowptr[a], self.rowptr[a + 1]) # indices a considerer dans colind et data

            for b in indices: # intervalle d'indices de nnz pour le dimension la plus imbriquee

                k = self.colind[b] # indices de la 3e dimension
                j = a % y # indices de la 2e dimensions
                i = a // y # indices de la 1re dimension
                matrix[i][j][k] = self.data[b] # reaffectation des nnz          

        return matrix.tolist()


    def __getitem__(self, k):
        '''
        :param k: iterable triple d'indices (i,j,k)
        :return: la valeur à la position (i,j,k)
        '''
        i, j, z = k # indices recherchés

        l =  len(self.rowptr)//self.n # nombre d'intervalle par rangée
        min = self.rowptr[l*i+j] 
        max = self.rowptr[l*i+j+1]-1

        # On cherche l'index de la 3e dimension dans colind.
        index = binarySearch(self.colind, z, self.rowptr[l*i+j], self.rowptr[l*i+j+1]-1)

        if index == -1:
            return 0
            
        return self.data[index]


#fonction récursive pour chercher un élément dans une séquence triée
def binarySearch(data, target, min=None, max=None):
    '''
    :param data: une liste triée
    :param target: la valeur cherchée
    :param min: indice de la borne inférieure de l'intervalle de recherche dans data
    :param max: indice de la borne supérieure de l'intervalle de recherche dans data
    :return: l'indice de target dans data ou -1 si c'est impossible.
    '''
    if min == None:
        min = 0
    if max == None:
        max = len(data)-1

    if min > max: # base case : no match
        return -1
    else:
        moy = (min+max)//2
        if target == data[moy]: # base case : on a un match 
            return moy
        elif target < data[moy]:
            return binarySearch(data, target, min, moy -1)
        else:
            return binarySearch(data, target, moy + 1, max)
