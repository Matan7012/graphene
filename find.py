from __future__ import print_function
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import skimage.filters
import skimage.color


# image = cv2.imread("try_photo.tif")
# img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
# hsv_color1 = np.asarray([130, 0, 0]) #light purple
# hsv_color2 = np.asarray([150, 255, 255]) #dark purple
#
# mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)
def calibration():
    return


def find_rgb(src, start_range, finish_range):
    hsv_color1 = np.asarray(start_range)  # light purple
    hsv_color2 = np.asarray(finish_range)  # dark purple

    mask = cv2.inRange(src, hsv_color1, hsv_color2)
    median = cv2.medianBlur(mask, 5)
    plt.imshow(median)
    plt.show()


def filter_and_smooth(grayimage):
    # Load image, convert to grayscale, Gaussian blur, Otsu's threshold
    blur = cv2.GaussianBlur(grayimage, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Filter using contour area and remove small noise
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 5500:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

    # Morph close and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # cv2.imshow('thresh', thresh)
    # plt.imshow(close)
    # plt.show()
    # cv2.waitKey()
    return close


def filter_gray(src, background):
    hsv_color1 = np.asarray(background - 2)  # light purple
    hsv_color2 = np.asarray(background + 2)  # dark purple
    gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray_image, hsv_color1, hsv_color2)
    median = cv2.medianBlur(mask, 5)
    image_without_background = abs(255 - median)
    filterd_image = filter_and_smooth(image_without_background)

    plt.imshow(filterd_image)
    plt.show()
    return filterd_image


def filter_rgb(src, background):
    """

    :param path:  image
    :param background:
    :return:
    """
    blue_channel = src[:, :, 0]
    green_channel = src[:, :, 1]
    red_channel = src[:, :, 2]
    threshold=[10,10,10]
    hsv_color1 = np.asarray(np.array(background) -np.array(threshold))
    hsv_color2 = np.asarray(np.array(background) +np.array(threshold))
    background_image = cv2.inRange(src, hsv_color1, hsv_color2)
    median = cv2.medianBlur(background_image, 5)
    # image_without_background = abs(255 - median)
    # print(image_without_background)
    # filterd_image = filter_and_smooth(median)
    plt.imshow(median)
    plt.show()
    return median


def histogramgray(path):
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray_image)
    # plt.show()

    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    max_location = np.where(histogram == histogram.max())
    listOfCoordinates = list(zip(max_location[0], max_location[1]))
    # plt.plot(histogram, color='k')
    # plt.show()
    # print(listOfCoordinates[0][0])
    return listOfCoordinates[0][0]
    # cv2.imshow('Gray image', gray_image)
    # cv2.waitKey(0)


def histogramrgb(path):
    image = cv2.imread(path)
    array_of_bgr_max_location = []
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        # calculte where is the max for b,g,r
        max_location = np.where(hist == hist.max())
        listOfCoordinates = list(zip(max_location[0], max_location[1]))
        array_of_bgr_max_location.append(listOfCoordinates[0][0])
    #     plt.plot(hist, color=col)
    #     plt.xlim([0, 256])
    # print(array_of_bgr_max_location)
    # plt.show()
    return array_of_bgr_max_location


def bgr2rgb(image):
    b, g, r = cv2.split(image)  # get b,g,r
    image_rgb = cv2.merge([r, g, b])  # switch it to rgb
    return image_rgb


def edge_detiction(path):
    # Read the original image
    img = cv2.imread(path)
    # Display original image

    figwidth, figheight = plt.rcParams['figure.figsize']
    fig, axes = plt.subplots(2, 2, figsize=(2 * figwidth, figheight))
    axes[0][0].imshow(bgr2rgb(img))

    # cv2.imshow('Original', img)
    # cv2.waitKey(0)
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    # Display Sobel Edge Detection Images
    # cv2.imshow('Sobel X', sobelx)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel Y', sobely)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    # cv2.waitKey(0)
    axes[0][1].imshow(sobelx, cmap='gray', vmin=0, vmax=255)
    axes[1][0].imshow(sobely, cmap='gray', vmin=0, vmax=255)
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=0.1, threshold2=20)  # Canny Edge Detection
    # Display Canny Edge Detection Image
    # cv2.imshow('Canny Edge Detection', edges)
    # cv2.waitKey(0)

    axes[1][1].imshow(edges, cmap='gray', vmin=0, vmax=255)

    plt.show()
    cv2.destroyAllWindows()
    return sobelx, sobely


if __name__ == '__main__':
    path = 'C:\\Users\\Owner\\PycharmProjects\\graphene\\try_photo.tif'
    value_of_background_grayscale = histogramgray(path)
    src = cv2.imread(path)
    plt.imshow(bgr2rgb(src))
    plt.show()
    value_of_background_bgr = histogramrgb(path)
    image_filterd_from_flake=filter_rgb(src, value_of_background_bgr)
    filterd_image = filter_gray(src, value_of_background_grayscale)
    #
    # image_without_background=src*filterd_image
    for row in range(len(image_filterd_from_flake)):
        for colum in range(len(image_filterd_from_flake[row])):
            if image_filterd_from_flake[row][colum] == 0 or filterd_image[row][colum]==0:
                src[row][colum] = [0, 0, 0]

    plt.imshow(bgr2rgb(src))
    plt.show()
    # start_range = [118, 164, 230]
    # finish_range = [123, 168, 235]
    # find_rgb(src, start_range, finish_range)

    # edge_detiction(path)
