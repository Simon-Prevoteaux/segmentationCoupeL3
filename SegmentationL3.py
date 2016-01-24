#!/usr/bin/python
# -*- coding: utf-8 -*-


import theano
import cPickle as pickle
import numpy as np
import crino
from crino.network import MultiLayerPerceptron
import scipy.io as sio
import scipy
import scipy.ndimage
from cv2 import imshow ,waitKey
from PyQt4 import QtGui, QtCore

import cv2
from math import cos, sin

from pylab import dot

import sys
import dicom

def normalise(image):
	"""
		Permet de normaliser l'image entre 0 et 1
	"""
	maxi=image.max()
	mini=image.min()
	imageNormalisee=np.zeros((image.shape[0],image.shape[1]))
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			imageNormalisee[i,j]=(image[i,j]-mini)/(maxi-mini).astype(float)
	return imageNormalisee

def humuscle(image):
	"""
		Permet de ne garder que ce qui correspond à la plage des muscles, ce qui dépasse étant fixé aux bornes de l'intervalle de la plage des muscles
	"""
	imageTemp=image.__deepcopy__(image)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if image[i,j]<-29:
				imageTemp[i,j]=-29
			elif image[i,j]>150:
				imageTemp[i,j]=150
	return imageTemp

def huall(image):

	imageTemp=image.__deepcopy__(image)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if image[i,j]<-190:
				imageTemp[i,j]=-190
			elif image[i,j]>150:
				imageTemp[i,j]=150
	return imageTemp

def get_masque(imageHU,seuilInf,seuilMax):
        """
        Effectue un seuillage sur une image en HU pour les valeurs compris entre seuilInf et seuilMax
        """
	masque=np.zeros(imageHU.shape)

	for i in range(imageHU.shape[0]-1): #on peut par exemple parcourir jusqu'au mini de la shape
		for j in range(imageHU.shape[0]-1):
			if imageHU[i,j]>seuilInf and imageHU[i,j]<seuilMax:
				masque[i,j]=1
			else:
				masque[i,j]=0
	return masque
        


def retirer_peau(imageBase):
	"""
		Permet de réaliser une érosion et ainsi de supprimer la peau du patient 
		on prend en compte toute la plage hu des tissus -190-150
		on effectue deux érosions consécutives
		ce qui ne rentre pas dans le masque de l'érosion est fixé à -190
	"""

	masque_muscle=np.zeros(imageBase.shape)
	image=huall(imageBase)
	masque_muscle[image>image.min()]=1

	rayon=2
	y,x=np.ogrid[-rayon:rayon+1,-rayon:rayon+1]
	mask=x**2+y**2<=rayon**2


	erosion=scipy.ndimage.morphology.binary_erosion(masque_muscle,structure=mask)
	masque_erosion=np.zeros(erosion.shape)
	masque_erosion[erosion]=1

	erosion=scipy.ndimage.morphology.binary_erosion(masque_erosion,structure=mask)
	masque_erosion[:,:]=0
	masque_erosion[erosion]=1


	image[np.invert(erosion)]=-190


	return image

def seuillage_80(masque,image,handle=False):
	"""
		Permet de retirer 20 pourcent des pixels les plus clairs.
	"""
	nb_pixel=sum(sum(masque))
	seuil=nb_pixel*20/100
	image=humuscle(image)
	if handle:
		print('nombre de pixel avant seuillage')
		print(nb_pixel)
		print('nombre de pixel a supprimer')
		print(seuil)

	if handle:
		imshow('masque initial',masque)
	k=0
	while(k<seuil):
		masque[np.unravel_index(np.argmax(image),image.shape)]=0
		image[np.unravel_index(np.argmax(image),image.shape)]=image.min()
		k+=1

	if handle:
		imshow('masque apres seuillage a 80%',masque)
		print('nombre de pixel apres seuillage')
		print(sum(sum(masque)))
	return masque


def recaler(imageARecaler, modele,masque_muscle,centre_lombaire_image_a_recaler,range_param,rec,handle=False):
	"""
		Permet d'effectuer le recalage de l'image par rapport au modele.

        :Parameters:
            imageARecaler
                L'image sur laquelle on veut effectuer une segmentation
            modele
            	Le modèle qui sert de modèle moyen et sur lequel on veut recaler l'imageTemp
            masque_muscle
            	Un masque binaire qui représente les muscles de imageARecaler
            centre_lombaire_image_a_recaler
            	Les coordonnées du centre de la lombaire de imageARecaler
            range_param
            	Les intervalles entre lesquels on va faire varier l'angle de recalage et les centres

        :return: Une image recalée par rapport au modèle.
	"""


	masque_modele=modele['shape']
	m=masque_modele.shape[0]
	n=masque_modele.shape[1]


	#faire un meilleur seuillage pour avoir un vrai masque que des muscles
	if(handle):
		imshow('image en entree du recalage',imageARecaler)
		imshow('masque du muscle',masque_muscle)
		imshow('masque du modele', masque_modele.astype(float))

	# a priori ça fonctionne meme si le masque du muscle du coup est pas netoyé ( lombaire etc ) :///

	masque_muscle=seuillage_80(masque_muscle,imageARecaler,handle=False)

	nSub=5
	angleMin=range_param[0,0]
	angleMax=range_param[0,1]
	tmin=range_param[1,0]
	tmax=range_param[1,1]

	centre_lombaire_modele=modele['center'][0]

	largeur_modele,hauteur_modele,centre_modele=infos_image(masque_modele)
	largeur_image,hauteur_image,centre_image=infos_image(masque_muscle)


	tfMat=np.array([[1,0,centre_modele[0]],[0,1,centre_modele[1]],[0,0,1]])

	tfMatInv=np.linalg.inv(tfMat)


    
	angles= np.linspace(angleMin,angleMax,nSub)
	T=np.linspace(tmin,tmax,nSub)

	score=np.zeros((nSub,nSub))
	final_warp = [[0]*nSub for i in range(nSub)]
	for i in range(nSub):
		ang=angles[i]
		for j in range(nSub):
			t=T[j]
			warp_init=np.zeros((3,3))

    		#--------------------recalage angle------------------------
			angr=ang/180*3.14
			warp_init=np.array([[cos(angr),-sin(angr),0],[sin(angr),cos(angr),0],[0,0,1]])
			warp_init=dot(dot(tfMat,warp_init),tfMatInv)
			final_warp[i][j]=warp_init

			
			#-----------------------------------
			mat_trans=np.linalg.inv(warp_init)
			temp=cv2.warpAffine(masque_modele,mat_trans[0:2,:],(n,m))
			#-----------------------
			

			
			#-----------------------recalage hauteur-largeur------------------------
			warp_init=np.eye(3)	

			largeur_modele,hauteur_modele,centre_modele=infos_image(temp)


			warp_init[0,0]=float(largeur_modele)/largeur_image
			warp_init[1,1]=float(hauteur_modele)/2+abs(centre_modele[1]-centre_lombaire_modele[1])
			warp_init[1,1]=float(warp_init[1,1])/(hauteur_image/2 + abs(centre_image[1]-centre_lombaire_image_a_recaler[1]))
			warp_init[1,1]=t*warp_init[1,1]+(1-t)*(float(hauteur_modele)/hauteur_image)


			warp_init=dot(dot(tfMat,warp_init),tfMatInv)
			final_warp[i][j]=dot(final_warp[i][j],warp_init)



			#----------------------recalage du centre-------------------------


			warp_init=np.eye(3)

			warp_init[0,2]=t*float(centre_lombaire_modele[0]-centre_lombaire_image_a_recaler[0]) + (1-t)*float(centre_modele[0]-centre_image[0])
			warp_init[1,2]=t*float(centre_lombaire_modele[1]-centre_lombaire_image_a_recaler[1]) + (1-t)*float(centre_modele[1]-centre_image[1])


			'''
			import pdb
			pdb.set_trace()
			'''

			final_warp[i][j]=dot(final_warp[i][j],warp_init)			
			warp_init=final_warp[i][j]

			#-------------------------
			mat_trans=np.linalg.inv(warp_init)
			temp=cv2.warpAffine(masque_modele,mat_trans[0:2,:],(n,m))
			#-------------------------------


			score[i,j]=sum(sum(np.logical_and(masque_muscle,temp)))

			
	imax,jmax=get_indice_max_matrice(score)

	
	matrice_recalage=final_warp[imax][jmax]
	angle_opti=angles[imax]
	t_opti=T[jmax]
	print('angle optimal :')
	print(angle_opti)

	matrice_temp=np.linalg.inv(matrice_recalage)
	modele_recale_temp=cv2.warpAffine(masque_modele,matrice_temp[0:2,:],(n,m))



	if rec>=2:
		temp=np.linalg.inv(matrice_recalage)
		return temp
	else:
		offset=(abs(angleMax-angleMin))/(2*nSub)
		offsetT=float(tmax-tmin)/(2*nSub)
		rec=rec+1
		range_param=np.array([[angle_opti-offset,angle_opti+offset],[max(0,t_opti-offsetT),min(1,t_opti+offsetT)]])
		matrice_recalage=recaler(imageARecaler, modele,masque_muscle,centre_lombaire_image_a_recaler,range_param,rec,handle=False)
		temp=np.linalg.inv(matrice_recalage)
		return temp

def infos_image(image):
        """
        Permet d'obtenir la largeur, hauteur et les coordonnées du centre d'un masque binaire
        """
	bounding_box=get_bouding_box(image)
	largeur=bounding_box[0,1]-bounding_box[0,0]
	hauteur=bounding_box[1,1]-bounding_box[1,0]
	centre=[bounding_box[0,0]+largeur/2,bounding_box[1,0]+hauteur/2]

	return [largeur, hauteur ,centre]

def get_indice_max_matrice(mat):
	"""
	Permet d'obtenir les indices i et j correspondant à la valeur max de la matrice
	"""
	maxi=mat[0,0]
	imax=0
	jmax=0
	for i in range(0,mat.shape[0]):
		for j in range(0,mat.shape[1]):
			if mat[i,j]>=maxi:
				imax=i
				jmax=j
				maxi=mat[i,j]

	return [imax,jmax]
	#return [jmax,imax]


def get_bouding_box(image):
	"""
		Permet d'obtenir la boite englobante minimum d'une image.
	"""
	label,num=scipy.ndimage.measurements.label(image)            
	sizes=scipy.ndimage.sum(image,label,range(num+1)).astype(int)
	mask_size=sizes<10 #si moins de 10 pixels on supprime la forme
	remove_pixel=mask_size[label]
	image[remove_pixel]=0

	#voir si ça change quelque chose
	image[image>0]=1

	parcourt=True
	i=0
	while(parcourt and i<image.shape[0]):
		if image[i,:].max()==1:
			y_min=i
			parcourt=False
		else:
			i+=1

	parcourt=True
	i=image.shape[0]-1
	while(parcourt and i>0):
		if image[i,:].max()==1:
			y_max=i
			parcourt=False
		else:
			i-=1


	parcourt=True
	j=0

	while(parcourt and j<image.shape[1]):
		if image[:,j].max()==1:
			x_min=j
			parcourt=False
		else:
			j+=1

	parcourt=True
	j=image.shape[1]-1
	while(parcourt and j>0):
		if image[:,j].max()==1:
			x_max=j
			parcourt=False
		else:
			j-=1

	return np.array([[x_min,x_max],[y_min,y_max]])


def find_lumbar_center(image):
	"""
		Permet de trouver le centre de la lombaire L3 ainsi qu'un masque de la L3
	"""

	masque_lombaire=np.zeros(image.shape)
	masque_lombaire[image>130]=1

	if (masque_lombaire.any()==1):

		rayon=2
		y,x=np.ogrid[-rayon:rayon+1,-rayon:rayon+1]
		mask_disk=x**2+y**2<=rayon**2


		label,num=scipy.ndimage.measurements.label(masque_lombaire)           
		sizes=scipy.ndimage.sum(image,label,range(num+1)).astype(int)
		mask_size=sizes<sizes.max() #vecteur de booléen
		remove_pixel=mask_size[label]
		masque_lombaire[remove_pixel]=0
		rows,col=np.where(masque_lombaire==1)
		centreL3_x=int(col.mean())
		centreL3_y=int(rows.mean())
		masque_lombaire[centreL3_y,centreL3_x]=1 # la ligne = y et la colonne = x
		masque_lombaire_plein=scipy.ndimage.morphology.binary_fill_holes(masque_lombaire.astype(int))
		return [centreL3_x,centreL3_y, masque_lombaire_plein]

	else:
		raise Exception('Pas de lombaire détectée.')

def rescale(data):
	"""
		Permet de passer d'une représentation en 16 bit de l'image à des valeurs en HU.
	"""
    # get numpy array as representation of image data
    arr = data.pixel_array.astype(np.float64)

    # pixel_array seems to be the original, non-rescaled array.
    # If present, window center and width refer to rescaled array
    # -> do rescaling if possible.
    if ('RescaleIntercept' in data) and ('RescaleSlope' in data):
        intercept = data.RescaleIntercept  # single value
        slope = data.RescaleSlope
        slopef=slope.__float__()
        interceptf=intercept.__float__()
        arr = arr*slopef+interceptf
        return arr

def retire_table(imageHU):
	"""
		Permet de retirer la table d'une seule image en unités HU
	"""
	imgTemp=np.zeros(imageHU.shape)

	for i in range(imgTemp.shape[0]):
		for j in range(imgTemp.shape[1]):
			if imageHU[i,j]>-190:
				imgTemp[i,j]=1
			else:
					imgTemp[i,j]=0

	label,num=scipy.ndimage.measurements.label(imgTemp)

	sizes=scipy.ndimage.sum(imgTemp,label,range(num+1)).astype(int)
	mask_size=sizes<sizes.max() #vecteur de booléen
	remove_pixel=mask_size[label]
	imageHU[remove_pixel]=-1024

def transform_numpy_arr(arrParam, window_center, window_width,
                           lut_min=0, lut_max=255):
	"""
		Look up table qui permet de passer des valeurs en HU à des valeurs sur 8bits entre 0 et 255 que l'on peut affichier facilement.
		:Parametre:
			window_center : valeur centrale sur laquelle on recentre les valeurs
			window_width : largeur de la fenetre
	"""
    arr=arrParam.__deepcopy__(arrParam)

    if np.isreal(arr).sum() != arr.size:
        raise ValueError

    # currently only support 8-bit colors
    if lut_max != 255:
        raise ValueError

    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)

    # LUT-specific array scaling
    # width >= 1 (DICOM standard)
    window_width = max(1, window_width)

    wc, ww = np.float64(window_center), np.float64(window_width)
    lut_range = np.float64(lut_max) - lut_min

    minval = wc - 0.5 - (ww - 1.0) / 2.0
    maxval = wc - 0.5 + (ww - 1.0) / 2.0

    min_mask = (minval >= arr)
    to_scale = (arr > minval) & (arr < maxval)
    max_mask = (arr >= maxval)

    if min_mask.any():
        arr[min_mask] = lut_min
    if to_scale.any():
        arr[to_scale] = ((arr[to_scale] - (wc - 0.5)) /
                         (ww - 1.0) + 0.5) * lut_range + lut_min
    if max_mask.any():
        arr[max_mask] = lut_max

    # round to next integer values and convert to unsigned int
    arr = np.rint(arr).astype(np.uint8)

    # return PGM byte-data string
    return arr


  
def segmentation_ioda(imageHU):
	"""
		Effectue la segmentation à partir d'une image en HU.
	"""


	model_mat=sio.loadmat('./ioda_256/mean_model_256.mat')
	modele=model_mat['model']
	donnees_modele=modele[0,0]
	

	image=imageHU.__deepcopy__(imageHU)

	image=retirer_peau(image) #on retire la peau de la coupe

	masque_muscle=get_masque(image,-29,150)
	masque_muscle_resize=cv2.resize(masque_muscle,dsize=(256,256)) #interpolation des valeurs entre 0 et 1
	masque_muscle_resize[masque_muscle_resize>0.5]=1
	masque_muscle_resize[masque_muscle_resize<=0.5]=0



	image=humuscle(image) #on ne garde que ce qui se trouve dans la plage des muscles

	image=normalise(image) #on la normalise

	image=scipy.misc.imresize(image,0.5) #reduction en image 256*256, fonctionne
	image=normalise(image) 






	# --------------- recalage ----------------------
	#/
	#|

	imageTemp = cv2.resize(imageHU,dsize=(256,256))
	centre_lombaire_image_a_recaler=find_lumbar_center(imageTemp)
	range_param=np.array([[-15,15],[0,1]])

	m=image.shape[0]
	n=image.shape[1]

	matrice_recalage=recaler(image,donnees_modele,masque_muscle_resize,centre_lombaire_image_a_recaler[0:2],range_param,0,handle=False)


	matrice_recalage_inverse=np.linalg.inv(matrice_recalage)

	image=cv2.warpAffine(image,matrice_recalage_inverse[0:2,:],(n,m))


	#|
	#\

	boxMat=sio.loadmat('./ioda_256/box_256.mat')
	box=boxMat['infos']['box'][0][0]

	minY=box[0][0]-1
	maxY=box[0][1]
	minX=box[1][0]-1
	maxX=box[1][1]

	image=image[minX:maxX,minY:maxY] 

	#--------------- initialisation du réseau --------------------------


	print('Initialisation du réseau ...')
	theano.config.floatX='float64' #initialisation
	trained_weights = pickle.load(open('./ioda_256/learned_params_256.pck'))
	nFeats=trained_weights['geometry'][0]
	feats=nFeats
	nLabels=nFeats
	nHidden=trained_weights['geometry'][1]
	geometry=[nFeats, nHidden, nHidden, nFeats] # on utilise la géométre du fichier qui contient les learned_params
	nn = MultiLayerPerceptron(geometry,outputActivation=crino.module.Sigmoid)
	nn.linkInputs(theano.tensor.matrix('x'), nFeats)
	nn.prepare(aPreparer=False) 
	nn.setParameters(trained_weights) 
	print('fin de l initialisation')

	# --------------------- segmentation ---------------------------
	print('Debut de la segmentation ...')

	x_test = np.zeros((1,nFeats))
	y_sortie = np.zeros((1,nFeats))
	x_test = np.asarray(x_test, dtype=theano.config.floatX) 
	x_test=image.reshape((1,feats),order='F')


	#---------------------------------forward---------------------------------

	y_sortie = nn.forward(x_test[0:1]) #on a un y estime entre 0 et 1
	y_sortie=normalise(y_sortie) #

	y_estim=np.zeros((157,229))
	y_estim=y_sortie.reshape((157,229),order='F') #marche avec le order='F'


	y_256=np.zeros((256,256))
	y_256[45:202,17:246]=y_estim 

	
	sortie_recalee=cv2.warpAffine(y_256,matrice_recalage[0:2,:],(n,m))

	y_512=scipy.misc.imresize(sortie_recalee,2.) 

	y_512=normalise(y_512)


	
	masque=np.zeros((512,512))
	masque=np.asarray(masque,dtype=int)
	masque[y_512<0.2183]=0
	masque[y_512>=0.2183]=1
	

	print('Fin de la segmentation')


	return masque


def main(filename):
	imageDicom=dicom.read_file(filename)
	imageHU=rescale(imageDicom)
	retire_table(imageHU)

	masque_muscle=segmentation_ioda(imageHU)
	imshow('masque issu de la segmentation',masque_muscle.astype(float))
	imageAff=transform_numpy_arr(imageHU,200,800)
	imshow('image a segmenter',imageAff)
	waitKey(0)


if __name__=='__main__':
		filename=sys.argv[1:][0]
		main(filename)
        
