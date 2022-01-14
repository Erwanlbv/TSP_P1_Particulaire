import PIL.Image
import numpy as np
import scipy.stats as stats

from variables_p2 import *
from Donnees_Moodle.filtrage_particulaire import *


def erreur_quadra(x_est, x_reel):
    return np.sqrt((x_est - x_reel).dot(x_est - x_reel) / T)


def reechantillonage(X, W):
    print('Réechantillonage')
    return np.random.choice(X, len(X), p=W), np.ones(len(X))/len(X)


def D_squared(hist1, hist2):
    return 1 - np.sum(np.sqrt(hist1*hist2))


def filtrage_particulaire_img():

    W = np.zeros((T, N))
    X = np.zeros((T, N))
    ests = []

    w_init = np.ones(N) / N

    prem_image, names, nb_images, folder_name = lecture_image()
    res_obs = selectionner_zone()
    #Format res : [x, y, longueur, largeur]

    long = res_obs[2]
    larg = res_obs[3]

    hist_obs = calcul_histogramme(prem_image, res_obs, Nb)[-1]

    prem_parts = np.random.multivariate_normal(res_obs[0:2], R, size=N)
    res_parts = []

    for i in range(len(prem_parts)):
        res_parts.append([prem_parts[i, 0], prem_parts[i, 1], long, larg])
    res_parts = np.array(res_parts)

    hists_pred = []
    for i in range(len(res_parts)):
        hists_pred.append(calcul_histogramme(prem_image, res_parts[i], Nb)[-1])
    hists_pred = np.array(hists_pred)

    prem_w = []
    for i in range(len(hists_pred)):
        prem_w.append(np.exp(-l * D_squared(hist_obs, hists_pred[i] * w_init[i])))
    prem_w = np.array(prem_w)

    print( 'Poids total des poids avant norm :' + str(np.sum(prem_w)))
    prem_w = prem_w / np.sum(prem_w)

    est = np.array([0, 0], dtype='float64')
    for i in range(len(prem_parts)):
        est += prem_parts[i] * prem_w[i]

    print(' \n Estimation : ' + str(est))
    print('Observation : ' + str(res_obs[0:2]))



filtrage_particulaire_img()
