# Import the needed packages
# Adapted from https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image


main_path = r'C:\Users\juliu_000\Pictures'
os.chdir(main_path)
files = os.listdir()
print(files)

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title):
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    fig = plt.figure(title)
    plt.suptitle("MSE:{mse_} , SSIM:{ssim_}".format(mse_ = m, ssim_ = s))

    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis('off')

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis('off')

    plt.show()

def min_image_size(image_A, image_B):
	image_one_size = Image.open(image_A)
	image_two_size = Image.open(image_B)

	width_one, height_one = image_one_size.size
	width_two, height_two = image_two_size.size

	widths = [width_one, width_two]
	heights = [height_one, height_two]

	min_width = min(widths)
	min_height = min(heights)

	return (min_width, min_height)

def min_image_resize(image_A, image_B):
	im_one = Image.open(image_A)
	im_two = Image.open(image_B)

	new_im_one = im_one.resize(min_image_size(image_A,image_B))
	new_im_two = im_two.resize(min_image_size(image_A,image_B))
	return new_im_one, new_im_two


def img_dff(image_A,image_B):
	if image_one.shape == image_two.shape:
		difference = cv2.subtract(image_one,image_two)
		cv2.imshow('Difference of A and B',difference)
		cv2.waitKey(0)
		non_zeros = cv2.countNonZero(difference)
		if non_zeros == 0:
			print('The images are the same')
		else:
			print('The images are not the same')
	return

im1, im2 = min_image_resize(files[0],files[1])

# im1.show()
# im2.show()

# image_one = cv2.imread(files[0])
# image_two = cv2.imread(files[1])

np_im1 = np.array(im1)
np_im2 = np.array(im2)

image_one = cv2.cvtColor(np_im1, cv2.COLOR_BGR2GRAY)
image_two = cv2.cvtColor(np_im2, cv2.COLOR_BGR2GRAY)

compare_images(image_one, image_two, 'Image One vs Image Two')
img_dff(image_one, image_two)