#------------------------------------------------------------------------------
#						02 Oct 2017 Thrusday 09:20 AM
#						Rahul Kumeriya
#------------------------------------------------------------------------------

import framework

import tkinter as tk
from tkinter import *
import numpy as np
import tkinter.messagebox
from matplotlib import pyplot as plt
import scipy.ndimage

import sys
import scipy.fftpack as fftim
from scipy.misc.pilutil import Image
import numpy
import math
from skimage import filters
from tkinter import ttk

from PIL import Image, ImageTk, ImageChops
import tkinter.filedialog
import cv2

from skimage.filters.thresholding import threshold_otsu
import skimage.exposure as imexp
from scipy.ndimage import label

from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import label
from skimage.measure import regionprops
import math

from skimage.feature import match_template
from matplotlib.figure import Figure

import xlsxwriter



class GUI(framework.GUIFramework):
	
	menuitems = (
		"""File- &Open Image/Ctrl+O/self.display_image,
				&Open webcam/Ctrl+W/self.webcam, 
				&Process Image/Ctrl+P/self.image_processing, Sep,
				&Exit/Alt+F4/self.quit""",
		'Edit- &Edit//self.inverseTransformation',
		"""Filters- &Mean Filter//self.meanFilter, 
			&Median Filter//self.medianFilter, 
			&Max Filter//self.maxFilter, 
			&Min Filter//self.minFilter, 
			&Sobel Filter//self.sobelFilter, 
			&Prewitt Filter//self.prewittFilter, 
			&Lapacian Filter//self.lapacianFilter, 
			Sep,  
			&Fourier Transform Filter//self.fourierTransformation,
			Sep,
			&Ideal LPF Filter//self.idealLPF,
			&ButterWorth LPF Filter//self.butterworth_LPF, 
			&Guassian LPF Filter//self.gaussian_LPF, 
			Sep,
			&Ideal HPF Filter//self.idealHPF,
			&ButterWorth HPF Filter//self.butterworthHPF, 
			&Guassian HPF Filter//self.gaussianHPF, 
			Sep,
			&Bandpass Filter//self.bandPassFilter """ ,
		'''Image Enhancement- &Inverse Transformation//self.inverseTransformation,
							&Power Law Transformation//self.powerLawTransforamtion,
							&Log Transformation//self.logTransformation,
							Sep,
							&Histogram//self.histogram,
							Sep,
							&Contrast Streching//self.contrastStreching''',
							
		'''Segmentation- &Ostu//self.ostu,
							&Adaptive Thresholding//self.adaptive_thresholding,
							&Reny Entropy//self.renyEntropy,
							&Water Segmentation//self.water_segmentation''',
		'''Morfological Operation- &Binary Dilation//self.binary_dilation,
							&Erosion//self.erosion,
							&Grayscale Dialtion Erosion Opening//self.grayScale_dilation_erosion_Opening,
							&Grayscale Dialtion Erosion Closing//self.grayScale_dilation_erosion_Closing,
							&Hit or Miss//self.hitOrMiss,
							&Theickening//self.thickening,
							&Skeletonization//self.skeletonization''',											
		'Report- &PDF//self.report_generator,&EXCEL//self.excel_generator',
		'About- About//self.about'
		)

	
#>>>>>>>>>>>>>>>   FILE MENU OPTIONS   <<<<<<<<<<<<<<<<<<<
	
	def open_image(self):
		image1 = Image.open(tkinter.filedialog.askopenfilename())
		image1 = image1.resize((600, 700), Image.ANTIALIAS)
		#image1 = Image.open('wRI61du.jpg')
		#image1 = Image.open('images.jpg')
		return image1

	def display_image(self):
		global image1
		image1 = Image.open(tkinter.filedialog.askopenfilename())
		image1 = image1.resize((600, 700), Image.ANTIALIAS)
		tkimage1 = ImageTk.PhotoImage(image1)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
		
	def webcam(self):
		cap = cv2.VideoCapture(0)

		while(True):
			#Capture frame-by-frame
			ret, frame = cap.read()
			ret = cap.set(4,240)
			
			#Our operations on the frame come here
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			#Display the resulting frame
			cv2.imshow('frame',gray)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
				
		# When everything done, release the capture
		cap.release()
		cv2.destroyAllWindows()

	def image_processing(self):
		
		img = cv2.imread("MB53.bmp",0)
		img = cv2.medianBlur(img,5)
		#img = img.resize((600, 700), Image.ANTIALIAS)
		ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
		th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
		cv2.THRESH_BINARY,11,2)
		th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
		cv2.THRESH_BINARY,11,2)

		titles = ['Original Image', 'Global Thresholding (v = 127)',
		'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
		images = [img, th1, th2, th3]

		for i in range(4):
			plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
			plt.title(titles[i])
			plt.xticks([]),plt.yticks([])

		plt.show()
	
		
	def meanFilter(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		#OPENING THE IMAGE AND CONVERTING TO GRAYSCALE
		#INITILILIZING THE FILTER OF SIZE 5 BY 5
		#TH FILTER IS DIVIDED BY 25
		#a = a.resize((600, 700), Image.ANTIALIAS)
		k = np.ones((5,5))/25
		#PERFORMING CONVOLUTION
		b = scipy.ndimage.filters.convolve(a , k)
		#b IS CONVERTED IMAGE FROM ndarry TO AN IMAGE
		im3 = scipy.misc.toimage(b)
		#SAVING THE IMAGE AS imageinverse_output.png in folder
		tkimage1 = ImageTk.PhotoImage(im3)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
	
	def medianFilter(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		b = scipy.ndimage.filters.median_filter(a, size=5, footprint = None,output=None, mode='reflect', cval=0.0, origin=0)
		b = scipy.misc.toimage(b)
		
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
	def maxFilter(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		b = scipy.ndimage.filters.maximum_filter(a, size=5, footprint = None,output=None, mode='reflect', cval=0.0, origin=0)
		b = scipy.misc.toimage(b)
		
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1

	def minFilter(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		b = scipy.ndimage.filters.minimum_filter(a, size=5, footprint = None,output=None, mode='reflect', cval=0.0, origin=0)
		b = scipy.misc.toimage(b)
		
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
	
	def sobelFilter(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		#SOBEL FILTER
		b = filters.sobel(a)
		b = scipy.misc.toimage(b)
		
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
	def prewittFilter(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		#PREWITT FILTER
		b = filters.prewitt(a, mask=None)
		b = scipy.misc.toimage(b)
		
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
		
	def lapacianFilter(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		b = scipy.ndimage.filters.laplace(a, mode='reflect')
		b = scipy.misc.toimage(b)
		
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
		
	def fourierTransformation(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		
		b = numpy.asarray(a)
		c = abs(fftim.fft2(b))
		d = fftim.fftshift(c)
		# converting the d to floating type and saving it
		# as fft1_output.raw in Figures folder
		im2= d.astype(float)
		# im2 is  converting from ndarray  to an image
		b = scipy.misc.toimage(im2)
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
		
	def idealLPF(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		
		# a is converted to an ndarray
		b = numpy.asarray(a)
		# performing FFT
		c = fftim.fft2(b)
		# shifting the Fourier frequency image
		d = fftim.fftshift(c)
		# intializing variables for convolution function
		M = d.shape[0]
		N = d.shape[1]
		# H is defined and
		# values in H are initialized to 1
		H = numpy.ones((M,N))
		center1 = M/2
		center2 = N/2
		d_0 = 30.0 # cut-off radius
		# defining the convolution function for ILPF
		for i in range(1,M):
			for j in range(1,N):
				r1 = (i-center1)**2+(j-center2)**2
				# euclidean distance from
				# origin is computed
				r = math.sqrt(r1)
				# using cut-off radius to eliminate
				# high frequency
				if r > d_0:
					H[i,j] = 0.0
		# converting H to an image
		H = scipy.misc.toimage(H)
		# performing the convolution
		con = d * H
		# computing the magnitude of the inverse FFT
		e = abs(fftim.ifft2(con))
		# e is converted from an ndarray to an image
		b = scipy.misc.toimage(e)
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1		
		
	def butterworth_LPF(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		# a is converted to an ndarray
		b = scipy.misc.fromimage(a)
		# performing FFT
		c = fftim.fft2(b)
		# shifting the Fourier frequency image
		d = fftim.fftshift(c)
		# intializing variables for convolution function
		M = d.shape[0]
		N = d.shape[1]
		# H is defined and
		# values in H are initialized to 1
		H = numpy.ones((M,N))
		center1 = M/2
		center2 = N/2
		d_0 = 30.0 # cut-off radius
		t1 = 1 # the order of BLPF
		t2 = 2*t1
		# defining the convolution function for ILPF
		for i in range(1,M):
			for j in range(1,N):
				r1 = (i-center1)**2+(j-center2)**2
				# euclidean distance from
				# origin is computed	
				r = math.sqrt(r1)
				# using cut-off radius to eliminate
				# high frequency
				if r > d_0:
					H[i,j] = 1/(1+ (r/d_0)**t1)
		# converting H to an image
		H = scipy.misc.toimage(H)
		# performing the convolution
		con = d * H
		# computing the magnitude of the inverse FFT
		e = abs(fftim.ifft2(con))
		# e is converted from an ndarray to an image
		im3 = scipy.misc.toimage(e)
		#SAVING THE IMAGE AS imageinverse_output.png in folder
		tkimage1 = ImageTk.PhotoImage(im3)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
	
	def gaussian_LPF(self):
		global image1
		a = ImageChops.invert(image1).convert('L')			
		# a is converted to an ndarray
		b = scipy.misc.fromimage(a)
		# performing FFT
		c = fftim.fft2(b)
		# shifting the Fourier frequency image
		d = fftim.fftshift(c)
		# intializing variables for convolution function
		M = d.shape[0]
		N = d.shape[1]
		# H is defined and
		# values in H are initialized to 1
		H = numpy.ones((M,N))
		center1 = M/2
		center2 = N/2
		d_0 = 30.0 # cut-off radius
		t1 = 2*d_0
		# defining the convolution function for ILPF
		for i in range(1,M):
			for j in range(1,N):
				r1 = (i-center1)**2+(j-center2)**2
				# euclidean distance from
				# origin is computed	
				r = math.sqrt(r1)
				# using cut-off radius to eliminate
				# high frequency
				if r > d_0:
					H[i,j] = math.exp(-r**2/t1**2)
		# converting H to an image
		#H = PIL.toimage(H)
		H = scipy.misc.toimage(H)

		# performing the convolution
		con = d * H
		# computing the magnitude of the inverse FFT
		e = abs(fftim.ifft2(con))
		# e is converted from an ndarray to an image
		f = scipy.misc.toimage(e)
		#SAVING THE IMAGE AS imageinverse_output.png in folder
		tkimage1 = ImageTk.PhotoImage(f)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
	def idealHPF(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		# a is converted to an ndarray
		b = scipy.misc.fromimage(a)
		# performing FFT
		c = fftim.fft2(b)
		# shifting the Fourier frequency image
		d = fftim.fftshift(c)
		# intializing variables for convolution function
		M = d.shape[0]
		N = d.shape[1]
		# H is defined and
		# values in H are initialized to 1
		H = numpy.ones((M,N))
		center1 = M/2
		center2 = N/2
		d_0 = 30.0 # cut-off radius
		# defining the convolution function for ILPF
		for i in range(1,M):
			for j in range(1,N):
				r1 = (i-center1)**2+(j-center2)**2
				# euclidean distance from
				# origin is computed
				
				r = math.sqrt(r1)

				# using cut-off radius to eliminate
				# high frequency
				if 0 < r < d_0:
					H[i,j] = 0.0
					
		# converting H to an image
		#H = PIL.toimage(H)
		H = scipy.misc.toimage(H)

		# performing the convolution
		con = d * H
		# computing the magnitude of the inverse FFT
		e = abs(fftim.ifft2(con))
		# e is converted from an ndarray to an image
		f = scipy.misc.toimage(e)
		#SAVING THE IMAGE AS imageinverse_output.png in folder
		tkimage1 = ImageTk.PhotoImage(f)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1

		
	def butterworthHPF(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		
		# a is converted to an ndarray
		b = scipy.misc.fromimage(a)
		# performing FFT
		c = fftim.fft2(b)
		# shifting the Fourier frequency image
		d = fftim.fftshift(c)
		# intializing variables for convolution function
		M = d.shape[0]
		N = d.shape[1]
		# H is defined and
		# values in H are initialized to 1
		H = numpy.ones((M,N))
		center1 = M/2
		center2 = N/2
		d_0 = 30.0 # cut-off radius
		t1 = 1 # the order of BLPF
		t2 = 2*t1
		# defining the convolution function for ILPF
		for i in range(1,M):
			for j in range(1,N):
				r1 = (i-center1)**2+(j-center2)**2
				# euclidean distance from
				# origin is computed
				
				r = math.sqrt(r1)

				# using cut-off radius to eliminate
				# high frequency
				if 0 < r < d_0:
					H[i,j] = 1/(1+ (r/d_0)**t2)

		# converting H to an image
		H = scipy.misc.toimage(H)

		# performing the convolution
		con = d * H

		# computing the magnitude of the inverse FFT
		e = abs(fftim.ifft2(con))

		# e is converted from an ndarray to an image
		f = scipy.misc.toimage(e)
		#SAVING THE IMAGE AS imageinverse_output.png in folder
		tkimage1 = ImageTk.PhotoImage(f)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
	
	def gaussianHPF(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		
		# a is converted to an ndarray
		b = scipy.misc.fromimage(a)
		# performing FFT
		c = fftim.fft2(b)
		# shifting the Fourier frequency image
		d = fftim.fftshift(c)
		# intializing variables for convolution function
		M = d.shape[0]
		N = d.shape[1]
		# H is defined and
		# values in H are initialized to 1
		H = numpy.ones((M,N))
		center1 = M/2
		center2 = N/2
		d_0 = 30.0 # cut-off radius
		t1 = 2*d_0
		# defining the convolution function for ILPF
		for i in range(1,M):
			for j in range(1,N):
				r1 = (i-center1)**2+(j-center2)**2
				# euclidean distance from
				# origin is computed
				r = math.sqrt(r1)
				# using cut-off radius to eliminate
				# high frequency
				if 0 < r < d_0:
					H[i,j] = 1 - math.exp(-r**2/t1**2)
		# converting H to an image
		#H = PIL.toimage(H)
		H = scipy.misc.toimage(H)
		# performing the convolution
		con = d * H
		# computing the magnitude of the inverse FFT
		e = abs(fftim.ifft2(con))
		# e is converted from an ndarray to an image
		f = scipy.misc.toimage(e)
		#SAVING THE IMAGE AS imageinverse_output.png in folder
		tkimage1 = ImageTk.PhotoImage(f)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
	
	def bandPassFilter(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		
		# a is converted to an ndarray
		b = scipy.misc.fromimage(a)
		# performing FFT
		c = fftim.fft2(b)
		# shifting the Fourier frequency image
		d = fftim.fftshift(c)
		# intializing variables for convolution function
		M = d.shape[0]
		N = d.shape[1]
		# H is defined and
		# values in H are initialized to 1
		H = numpy.zeros((M,N))
		center1 = M/2
		center2 = N/2
		d_0 = 30.0 # minimum cut-off radius
		d_1 = 50.0 # maximum cut-off radius
		# defining the convolution function for bandpass
		for i in range(1,M):
			for j in range(1,N):
				r1 = (i-center1)**2+(j-center2)**2
				# euclidean distance from
				# origin is computed
				r = math.sqrt(r1)
				# using cut-off radius to create
				# the band or annulus
				if r > d_0 and r < d_1:
					H[i,j] = 1.0
		# converting H to an image
		#H = PIL.toimage(H)
		H = scipy.misc.toimage(H)
		# performing the convolution
		con = d * H
		# computing the magnitude of the inverse FFT
		e = abs(fftim.ifft2(con))
		# e is converted from an ndarray to an image
		f = scipy.misc.toimage(e)
		#SAVING THE IMAGE AS imageinverse_output.png in folder
		tkimage1 = ImageTk.PhotoImage(f)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
	
#############   FILTERS END ###################################

####################   IMAGE ENHANCEMENT #########################

	
	def inverseTransformation(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		# im IS CONVETED TO AN ndarray
		im1 = scipy.misc.fromimage(a)
		#PERFORM THE INVERSE OPERATION
		im2 = 255 - im1
		# im2 IS CONVERTED FROM AN ndarray TO AN IMAGE
		im3 = scipy.misc.toimage(im2)
		#SAVING THE IMAGE AS imageinverse_output.png in folder
		tkimage1 = ImageTk.PhotoImage(im3)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
	
	def powerLawTransforamtion(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		# im IS CONVETED TO AN ndarray
		b = scipy.misc.fromimage(a)
		#GAMMA IS INITILIZED
		gamma = 0.5
		#b is CONVERTED TO TYPE FLOAT
		b1 = b.astype(float)
		# MAXIMUM VALUE IN b1 is DETERMINED
		b3 = numpy.max(b1)
		#b1 is normalized
		b2 = b1/b3
		#GAMMA-CORRECTION EXPONENT IS COMPUTED
		b3 = numpy.log(b2)*gamma
		#GAMMA CORRECTION IS PERFORMED
		c = numpy.exp(b3)*255.0
		# c is converted to type int
		c1 = c.astype(int)
		# c1 IS CONVERTED FROM AN ndarray TO AN IMAGE
		d = scipy.misc.toimage(c1)
		#DISPLAYING THE IMAGE
		tkimage1 = ImageTk.PhotoImage(d)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1

		
	def logTransformation(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		# im IS CONVETED TO AN ndarray
		b = scipy.misc.fromimage(a)
		#b is CONVERTED TO TYPE FLOAT
		b1 = b.astype(float)
		# MAXIMUM VALUE IN b1 is DETERMINED
		b2 = numpy.max(b1)
		#PERFORMING THE LOG TRANSFORMATION
		c = (255.0*numpy.log(1+b1))/numpy.log(1+b2)
		# c is converted to type int
		c1 = c.astype(int)
		# c1 IS CONVERTED FROM AN ndarray TO AN IMAGE
		d = scipy.misc.toimage(c1)
		#DISPLAYING THE IMAGE
		tkimage1 = ImageTk.PhotoImage(d)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
	def histogram(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		# img IS CONVETED TO AN ndarray
		img1 = scipy.misc.fromimage(a)
		#2D ARRAY IS CONVERTED TO AN 1D
		fl = img1.flatten()
		#HISTOGRAM and THE BINS OF THE IMAGE ARE COMPUTED
		hist, bins=np.histogram(img1,256,[0,255])
		#CUMULATIVE DISTRBUTION FUNCTION IS COMPUTED
		cdf = hist.cumsum()
		#PLACES WHERE CDF=0 IS MARKED OR IGNORED AND STORED IN cdf_m
		cdf_m = np.ma.masked_equal(cdf,0)
		#HISTOGRAM EQUALIZATION IS PERFORMED
		num_cdf_m = (cdf_m - cdf_m.min())*255
		den_cdf_m = (cdf_m.max() - cdf_m.min())
		cdf_m = num_cdf_m/den_cdf_m
		# the masked places in cdf_m are now 0
		cdf = np.ma.filled(cdf_m,0).astype('uint8')
		# cdf values are assigned in the flattened array
		im2 = cdf[fl]
		# im2 is 1D so we use reshape command to
		# make it into 2D
		im3 = np.reshape(im2,img1.shape)
		# converting im3 to an image
		im4 = scipy.misc.toimage(im3)
		tkimage1 = ImageTk.PhotoImage(im4)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
	def contrastStreching(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		# im IS CONVETED TO AN ndarray
		im1 = scipy.misc.fromimage(a)
		#FINDING THE MAXIMU AND MINUMUM PIXEL VALUE
		b = im1.max()
		a = im1.min()
		#ONVERTING im1 TO FLOAT
		c = im1.astype(float)
		#CONTRAST STRECHING TRANSFORMATION
		im2 = 255*(c-a)/(b-a)
		# im2 is  converting from ndarray  to an image
		im3 = scipy.misc.toimage(im2)

		# ...and then to ImageTk format
		tkimage1 = ImageTk.PhotoImage(im3)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
				
#####################  IMAGE ENHANCEMENT ENDS #####################

#%%%%%%%%%%%%%%%%   SEGENTATION %%%%%%%%%%%%%%%%%%%%%%

	def ostu(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		# opening the image and converting it to grayscale
		# a is converted to an ndarray
		a = scipy.misc.fromimage(a)
		# performing Otsu
		#===============================================================================================================
		#																												#
		# 	'threshold_adaptive', 'threshold_isodata', 'threshold_li', 'threshold_local', 'threshold_mean', 			#
		#	'threshold_minimum', 'threshold_niblack', 'threshold_otsu', 'threshold_sauvola', 'threshold_triangle', 		#
		#	'threshold_yen', 'try_all_threshold', 																		#
		#																												#
		#==============================================================================================================
		thresh = threshold_otsu(a)   
		# pixels with intensity greater than
		# theshold are kept
		b = a > thresh
		# b is converted from ndimage to
		b = scipy.misc.toimage(b)
		tkimage1 = ImageTk.PhotoImage(b)
		
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
				
	def adaptive_thresholding(self):
		global image1
		a = ImageChops.invert(image1).convert('L')
		# a is converted to an ndarray
		a = scipy.misc.fromimage(a)

		# performing adaptive thresholding
		b = filters.threshold_adaptive(a,41,offset = 10)
		#b = filters.threshold_local(a,41,offset = 10)
		
		# b is converted from an ndarray to an image
		b = scipy.misc.toimage(b)
		tkimage1 = ImageTk.PhotoImage(b)	
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
	def renyEntropy(self):
		def renyi_seg_fn(im,alpha):
			hist = imexp.histogram(im)

			# Convert all values to float
			hist_float = [float(i) for i in hist[0]]

			# compute the pdf
			pdf = hist_float / numpy.sum(hist_float)
			
			# compute the cdf
			cumsum_pdf = numpy.cumsum(pdf)
			
			s = 0
			e = 244 # assuming 8 bit image
			scalar = 1.0/(1-alpha)

			# A very small value to prevent
			# division by zero
			eps = np.spacing(1)

			rr = e-s

			# The second parentheses is needed because
			# the parameters are tuple
			h1 = np.zeros((rr,1))
			h2 = np.zeros((rr,1))
			# the following loop computes h1 and h2
			# values used to compute the entropy
			
			for ii in range(1,rr):
				iidash = ii+s
				temp1 = np.power(pdf[1:iidash]/cumsum_pdf[iidash],scalar)
				h1[ii] = np.log(np.sum(temp1)+eps)
				temp2 = np.power(pdf[iidash+1:255]/(1-cumsum_pdf[iidash]),scalar)
				h2[ii] = np.log(np.sum(temp2)+eps)
			
			
			T = h1+h2
			# Entropy value is calculated
			T = -T*scalar
			# location where the maximum entropy
			# occurs is the threshold for the renyi entropy
			location = T.argmax(axis=0)
			# location value is used as the threshold
			thresh = location
			return thresh

		# Main program
		# opening the image and converting it to grayscale

		global image1
		a = ImageChops.invert(image1).convert('L')
		# a is converted to an ndarray
		a = scipy.misc.fromimage(a)
		# computing the threshold by calling the function
		thresh = renyi_seg_fn(a,2)
		b = a > thresh
		# b is converted from an ndarray to an image
		b = scipy.misc.toimage(b)
		# saving the image as renyi_output.png
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
	def water_segmentation(self):
		global image1	
		a = ImageChops.invert(image1).convert('L')
		# from skimage.morphology import label
		# opening the image and converting it to grayscale
		a = cv2.imread('wRI61du.jpg')
		# covnerting image from color to grayscale
		a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
		# thresholding the image to obtain cell pixels
		thresh,b1 = cv2.threshold(a, 0, 255,
		cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		# since Otsus method has over segmented the image
		# erosion operation is performed
		b2 = cv2.erode(b1, None,iterations = 2)
		# distance transform is performed
		dist_trans = cv2.distanceTransform(b2, 2, 3)
		# thresholding the distance transform image to obtain
		# pixels that are foreground
		thresh, dt = cv2.threshold(dist_trans, 1,255, cv2.THRESH_BINARY)
		# performing labeling
		#labelled = label(b, background = 0)
		labelled, ncc = label(dt)
		# labelled is converted to 32-bit integer
		labelled = labelled.astype(numpy.int32)
		# performing watershed
		cv2.watershed(a, labelled)
		# converting the ndarray to image
		dt1 = scipy.misc.toimage(labelled)

		tkimage1 = ImageTk.PhotoImage(dt1)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

###################   MORFOLOGICAL OPERATIONS ####################

	def binary_dilation(self):
		global image1	
		im = ImageChops.invert(image1).convert('L')
		b = scipy.ndimage.morphology.binary_dilation(im,iterations=5)
		# converting b from an ndarray to an image
		b = scipy.misc.toimage(b)
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1

	def erosion(self):
		global image1	
		a = ImageChops.invert(image1).convert('L')
		b = scipy.ndimage.morphology.binary_erosion(a,iterations=25)
		b = scipy.misc.toimage(b)
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1

	def grayScale_dilation_erosion_Opening(self):
		global image1	
		a = ImageChops.invert(image1).convert('L')
		# defining the structuring element
		s = [[0,1,0],[1,1,1], [0,1,0]]
		# performing the binary opening for 5 iterations
		b = scipy.ndimage.morphology.binary_opening(a, structure=s,iterations=5)
		# b is converted from an ndarray to an image
		b = scipy.misc.toimage(b)
		# displaying the image
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
	def grayScale_dilation_erosion_Closing(self):
		global image1	
		a = ImageChops.invert(image1).convert('L')
		# defining the structuring element
		s = [[0,1,0],[1,1,1], [0,1,0]]
		# performing the binary opening for 5 iterations
		b = scipy.ndimage.morphology.binary_closing(a, structure=s,iterations=5)
		# b is converted from an ndarray to an image
		b = scipy.misc.toimage(b)
		# displaying the image
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		

	def hitOrMiss(self):
		global image1	
		a = ImageChops.invert(image1).convert('L')	
		structure1 = np.array([[1, 1, 0], [1, 1, 1],[1, 1, 1]])
		# performing the binary hit-or-miss
		b = scipy.ndimage.morphology.binary_hit_or_miss(a,structure1=structure1)
		# b is converted from an ndarray to an image
		b = scipy.misc.toimage(b)
		
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
	def thickening(self):
		global image1	
		a = ImageChops.invert(image1).convert('L')	
		structure1 = np.array([[1, 1, 0], [1, 1, 1],[1, 1, 1]])
		s = [[0,1,0],[1,1,1], [0,1,0]]
		structure1=structure1-s
		# performing the binary hit-or-miss
		b = scipy.ndimage.morphology.binary_hit_or_miss(a,structure1=structure1)
		# b is converted from an ndarray to an image
		b = scipy.misc.toimage(b)
		
		tkimage1 = ImageTk.PhotoImage(b)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1

	#Skeletonization

	def skeletonization(self):
		global image1	
		a = ImageChops.invert(image1).convert('L')	
		a = scipy.misc.fromimage(a)/ numpy.max(a)
		# performing skeletonization
		b = skeletonize(a)
		# converting b from an ndarray to an image
		c = scipy.misc.toimage(b)	
		tkimage1 = ImageTk.PhotoImage(c)
		panelA.configure(image=tkimage1)
		panelA.image = tkimage1
		
#################################################################

################## IMAGE MESAUREMTNT   ##################
	
	def quit(self):
		import sys;sys.exit()

		
#######################  REPORT GENERATOR ###################

	def report_generator(self):
		######## for report
		import time
		from reportlab.lib.enums import TA_JUSTIFY
		from reportlab.lib.pagesizes import letter
		from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
		from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
		from reportlab.lib.units import inch


		doc = SimpleDocTemplate("output/report.pdf",pagesize=letter,
								rightMargin=72,leftMargin=72,
								topMargin=72,bottomMargin=18)
		Story=[]
		logo = "logo.jpg"
		magName = "DGIBR"
		issueNum = "v1.0"
		subPrice = "500000.00"
		limitedDate = "30/11/2017"

		 
		formatted_time = time.ctime()
		full_name = "Mr. X"
		address_parts = ["Distrubuted location of X-City", "X, Provinance"]
		 
		im = Image(logo, 2*inch, 2*inch)
		Story.append(im)
		 
		styles=getSampleStyleSheet()
		styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
		ptext = '<font size=12>%s</font>' % formatted_time
		 
		Story.append(Paragraph(ptext, styles["Normal"]))
		Story.append(Spacer(1, 12))
		 
		# Create return address
		ptext = '<font size=12>%s</font>' % full_name
		Story.append(Paragraph(ptext, styles["Normal"]))       
		for part in address_parts:
			ptext = '<font size=12>%s</font>' % part.strip()
			Story.append(Paragraph(ptext, styles["Normal"]))   
		 
		Story.append(Spacer(1, 12))
		ptext = '<font size=12>Dear %s:</font>' % full_name.split()[0].strip()
		Story.append(Paragraph(ptext, styles["Normal"]))
		Story.append(Spacer(1, 12))
		 
		ptext = '<font size=12>We would like to welcome you to deal base for %s Application \
				We will provide the %s version fully sold basis of the excellent introductory price of $%s. Please respond by\
				%s.</font>' % (magName,
								issueNum,
								subPrice,
								limitedDate
								)
		Story.append(Paragraph(ptext, styles["Justify"]))
		Story.append(Spacer(1, 12))
		 
		 
		ptext = '<font size=12>Thank you very much and we look forward to serving you.</font>'
		Story.append(Paragraph(ptext, styles["Justify"]))
		Story.append(Spacer(1, 12))
		ptext = '<font size=12>Sincerely,</font>'
		Story.append(Paragraph(ptext, styles["Normal"]))
		Story.append(Spacer(1, 48))
		ptext = '<font size=12>Raahool</font>'
		Story.append(Paragraph(ptext, styles["Normal"]))
		Story.append(Spacer(1, 12))
		doc.build(Story)
		
#<<<<<<<<<<<<<<<<<<<<<<< EXCEL FILE GERNERATION >>>>>>>>>>>>>>>>>>>>>>>>>


	def excel_generator(self):
		# Create an new Excel file and add a worksheet.
		workbook = xlsxwriter.Workbook('output/report.xlsx')
		worksheet = workbook.add_worksheet()

		# Widen the first column to make the text clearer.
		worksheet.set_column('A:A', 20)

		# Add a bold format to use to highlight cells.
		bold = workbook.add_format({'bold': True})

		# Write some simple text.
		worksheet.write('A1', 'GIBDRT')

		# Text with formatting.
		worksheet.write('A2', 'GEL IMAGE BAND DETECTION AND REPORTING TOOL', bold)

		# Write some numbers, with row/column notation.
		worksheet.write(2, 0, 123)
		worksheet.write(3, 0, 123.456)
		
		logo = "logo.jpg"
		# Insert an image.
		worksheet.insert_image('B5', 'logo.jpg')

		workbook.close()

#>>>>>>>>>>>>>>>>>>>>>>>   ABOUT MENU OPTION <<<<<<<<<<<<<<<<<<<<<<<<<<
	
	def about(self):
		tkinter.messagebox.showinfo("About","Author: Raahool\n\nContact: rahulkumeriya@gmail.com \n\nMob: +91 9XXX 9696 XX")



if __name__ == '__main__':
	root= tk.Tk()
	app = GUI(root)
	root.title("GIBDRT")
	root.wm_iconbitmap('TrayIconEOO.ico')#changing the default icon
	root.geometry('1200x600+150+200')
	
	image1 = Image.open(tkinter.filedialog.askopenfilename())
	#image1 = Image.open('wRI61du.jpg')
	#image1 = Image.open('Kfg3Opb.jpg')
	image1 = image1.resize((600, 700), Image.ANTIALIAS)
	tkimage1 = ImageTk.PhotoImage(image1)
	
	panelA = tk.Label(root, image=tkimage1)
	panelA.pack(side='top', fill='both', expand='yes')
	# save the panel's image from 'garbage collection'
	panelA.image = tkimage1


	root.mainloop()
