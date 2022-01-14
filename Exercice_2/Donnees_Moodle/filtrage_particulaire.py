import matplotlib
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import matplotlib.image as mpimg
from PIL import Image
import math
from scipy.stats import norm
from numpy.random import random
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import os



def multinomial_resample(weights):
    weights=weights.T
    cumulative_sum = np.cumsum(weights)
    #cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    #print ( np.searchsorted(cumulative_sum, random(len(weights))))
    return np.searchsorted(cumulative_sum, random(len(weights)))


def lecture_image() :

    SEQUENCE = 'Donnees_Moodle/sequences/sequence1/'
    #charge le nom des images de la séquence 1
    filenames = os.listdir(SEQUENCE)
    T = len(filenames)
    #charge la premiere image dans ’im’
    tt = 0

    im=Image.open((str(SEQUENCE)+str(filenames[tt])))
    plt.imshow(im)
    
    return im, filenames, T, 'sequence/sequence1'


def selectionner_zone():

    #lecture_image()
    print("Cliquer 4 points dans l'image pour definir la zone a suivre.")
    zone = np.zeros([2,4])
 #   print(zone))
    compteur=0
    while(compteur != 4):
        print(compteur)
        res = plt.ginput(1)
        a=res[0]
        #print(type(a)))
        zone[0,compteur] = a[0]
        zone[1,compteur] = a[1]   
        plt.plot(a[0],a[1],marker='X',color='red') 
        compteur = compteur+1 

    #print(zone)
    newzone = np.zeros([2,4])
    newzone[0, :] = np.sort(zone[0, :]) 
    newzone[1, :] = np.sort(zone[1, :])
    
    zoneAT = np.zeros([4])
    zoneAT[0] = newzone[0,0]
    zoneAT[1] = newzone[1,0]
    zoneAT[2] = newzone[0,3]-newzone[0,0] 
    zoneAT[3] = newzone[1,3]-newzone[1,0] 
    #affichage du rectangle
    #print(zoneAT)
    xy=(zoneAT[0],zoneAT[1])
    rect=ptch.Rectangle(xy,zoneAT[2],zoneAT[3],linewidth=3,edgecolor='red',facecolor='None') 
    #plt.Rectangle(zoneAT[0:1],zoneAT[2],zoneAT[3])
    currentAxis = plt.gca()
    currentAxis.add_patch(rect)
    plt.show(block=False)
    return zoneAT


def rgb2ind(im,nb) :
    #nb = nombre de couleurs ou kmeans qui contient la carte de couleur de l'image de référence
    
    image=np.array(im,dtype=np.float64)/255
    w,h,d=original_shape=tuple(image.shape)
    image_array=np.reshape(image,(w*h,d))
    image_array_sample=shuffle(image_array,random_state=0)[:1000]
    #print(image_array_sample.shape)
   # print(type(image_array))
    if type(nb)==int :
        kmeans=KMeans(n_clusters=nb,random_state=0).fit(image_array_sample)
    else :
        kmeans=nb
            
    labels=kmeans.predict(image_array)
    #print(labels)
    image=recreate_image(kmeans.cluster_centers_,labels,w,h)
    #print(image)
    return(Image.fromarray(image.astype('uint8')),kmeans)


def recreate_image(codebook,labels,w,h):
    d=codebook.shape[1]
    #image=np.zeros((w,h,d))
    image=np.zeros((w,h))
    label_idx=0
    for i in range(w):
        for j in range(h):
            #image[i][j]=codebook[labels[label_idx]]*255
            image[i][j]=labels[label_idx]
            #print(image[i][j])
            label_idx+=1

    return image


def calcul_histogramme(im, zoneAT, Nb):

    box=(zoneAT[0],zoneAT[1],zoneAT[0]+zoneAT[2],zoneAT[1]+zoneAT[3])

    littleim = im.crop(box)

    new_im, kmeans = rgb2ind(littleim,Nb)
    histogramme =np.asarray(new_im.histogram())

    histogramme = histogramme/np.sum(histogramme)

    return new_im, kmeans, histogramme