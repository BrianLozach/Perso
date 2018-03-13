param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}


def customGridSearch(listeHyperParam, nbFold):
    for hyperParam in listeHyperParam:
        print(hyperParam)
    return nbFold


customGridSearch(param_grid, 2)
