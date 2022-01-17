import matplotlib.pyplot as plt

from main_p1 import *
from variables_p1 import *


def question_3():
    traj = creer_trajec(T, Q)
    obs = creer_observation(traj, R)
    obs_2 = creer_observation(traj, R)

    fig = plt.figure(figsize=(13, 7))
    fig.suptitle("Création et Bruitage de la trajectoire" + '\n' + 'R = ' + str(R) + ', Q = ' + str(Q), fontsize=14)

    plt.plot(traj, label='Trajectoire réelle', color='blue', alpha=0.7)
    plt.scatter(range(T), obs, label='Observations', color='orange', alpha=0.3)
    plt.scatter(range(T), obs_2, label='Observations_2', color='red', alpha=0.3)

    plt.legend()
    fig.show()


def question_6():
    traj = creer_trajec(T, Q)
    obs = creer_observation(traj, Q)
    w_seuil = .5

    W = 1/N * np.ones((T, N))
    X = np.random.normal(obs[0], Q, size=(T, N)) # On génère les premières particlues aléatoirement autour de la première observation.
    x_est = [W[0].dot(X[0])]
    pos_ree = np.zeros(T)

    for i in range(1, N):
        X[i], W[i], pos_ree[i] = filtrage_particulaire(W[i-1], X[i-1], obs[i], Q, R, x_est, i, T, w_seuil)

    print("Valeurs des particules : ")
    print(X[-2:])
    print('Valeur des poids : ')
    print(W[-2:])
    print('Valeurs des estimations : ')
    print(x_est)
    print('Nombre de rééchantillonages : ' + str(np.sum(pos_ree)))

    fig = plt.figure(figsize=(13, 7))
    fig.suptitle('Filtrage particicluaire avec rééchantillonage \n' + 'R : ' + str(R) + ' Q : ' + str(Q) + ' Seuil : ' + str(w_seuil) +
                 '\n Erreur quadratique moyenne : ' + str(erreur_quadra(x_est, traj)))

    plt.plot(traj, label='Trajectoire réelle', color='blue', alpha=0.7)
    plt.plot(x_est, label='Trajectoire estimée', color='orange', alpha=0.7)
    plt.scatter(range(T), obs, label='Observations', color='red', alpha=0.3)
    #plt.scatter(range(T), pos_ree, color='green', alpha=0.5)

    plt.legend()
    plt.show()


#question_3()
question_6()





