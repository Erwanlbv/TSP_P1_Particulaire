import numpy as np
import scipy.stats as stats

from variables_p1 import *

#Question 1:
#Du fait du X**2, on ne sait pas si la valeur du X est positive ou négative


def f(x_prec, n): #Chaîne non homogène (merde)
    return 0.5*x_prec + 25*x_prec/(1 + x_prec**2) + 8*np.cos(1.2*n)


def g(x_n):
    return x_n**2 / 20


def erreur_quadra(x_est, x_reel):
    return np.sqrt((x_est - x_reel).dot(x_est - x_reel) / T)


def creer_trajec(T, Q):
    result = [np.random.rand()]
    for i in range(1, T):
        result.append(f(result[-1], i) + np.random.normal(0, Q))
    return np.array(result)


def creer_observation(traj, R):
    result = [g(traj[0]) + np.random.normal(0, Q)]
    for i in range(1, len(traj)):
        result.append(g(traj[i]) + np.random.normal(0, R))
    return np.array(result)


def creer_obs_vect(traj, R):
    return g(traj) + np.random.normal(0, R, len(traj))


def reechantillonage(X, W):
    print('Réechantillonage')
    return np.random.choice(X, len(X), p=W), np.ones(len(X))/len(X)


def filtrage_particulaire(W, X, y_n, Q, R, estimations, n, T, w_seuil):  # X = [[particules t0], [particules t1], .. [particules t]], idem W

    print('\n \n -------------------- Itération numéro :  ' + str(n) + ' --------------- \n \n')
    ree = False
    new_part = np.random.normal(f(X, n), Q, T)

    #for i in range(T):
    #    new_w.append(stats.norm.pdf(y_n, g(new_part[i]), R)*W[-1])
    #print('-----' + str(new_w))

    #Version vectorielle :
    new_w = stats.norm.pdf(y_n, g(new_part), R) * W[-1]

    #Normalisation
    new_w = np.array(new_w)/np.sum(new_w)
    print((np.sum(new_w)))

    #Estimation
    est = new_part.dot(new_w)
    estimations.append(est)

    print("----- Valeur de l'observation : " + str(y_n))
    print("Valeur de l'estimation : " + str(est))

    #Réechantillonage si un poids est supérieur à w_seuil
    if (new_w >= w_seuil).any():
        ree = True
        new_part,  new_w = reechantillonage(new_part, new_w)

    return new_part, new_w, ree



