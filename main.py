from skimage.io import imread, imshow, imsave
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import math
import scipy.fftpack as fp
from skimage.color import rgb2gray
from skimage import img_as_ubyte, img_as_float


def print_img_params(img):
    print(f'{img.shape} - shape')
    print(f'{img.min()}, {img.max()}  - min, max')
    print(f'{img.dtype}  - datatype\n')


def create_kernel(sigma, half):
    """Returns the kernel for the Gaussian filter with the sigma parameter.

            Args:
                    sigma (float): the parameter of the Gaussian sigma spread
                    half (int): half the width of the filter kernel (radius)

            Returns:
                    kernel (list): filter kernel
    """
    k = half * 2 + 1

    kernel = []
    for i in reversed(range(-half, half + 1)):
        for j in range(-half, half + 1):
            kernel.append((j, i))
    kernel = (np.array(kernel).reshape(k, k, 2)).tolist()
    for i in range(k):
        for j in range(k):
            kernel[i][j] = 1 / (2 * math.pi * sigma ** 2) * math.exp(- (
                kernel[i][j][0] ** 2 + kernel[i][j][1] ** 2) / (2 * sigma ** 2))
    sum = np.array(kernel).sum()
    for i in range(k):
        for j in range(k):
            kernel[i][j] = kernel[i][j] / sum

    return kernel


def mirror(seq, half):
    output = list(seq[::-1][-half:])
    output.extend(seq[:])
    output.extend(seq[-1:-(half + 1):-1])
    return output


def mirroring_border(arr, half):
    arr_mirrored = mirror([mirror(sublist, half) for sublist in arr], half)
    arr_mirrored = np.array(arr_mirrored)
    return arr_mirrored


def make_fourier_transform(img):
    freq_img = fp.fft2(rgb2gray(img))
    freq = np.log(1 + np.abs(fp.fftshift(freq_img)))
    return freq


def create_gaussian_pyramid(img, sigma, n_gauss_layers):

    half = round(3 * sigma)
    kernel = create_kernel(sigma, half)

    list_img_a_gauss = [img]
    list_img_a_gauss_fourier = [make_fourier_transform(img)]
    img_cycle = img.copy()
    for _ in range(n_gauss_layers - 1):
        r, g, b = np.dsplit(img_cycle, 3)
        r_mirrored, g_mirrored, b_mirrored = [
            mirroring_border(i, half) for i in (r, g, b)]
        r_gauss, g_gauss, b_gauss = [(convolve2d(i[:, :, 0], kernel, mode='valid')).astype('uint8') for i in
                                     (r_mirrored, g_mirrored, b_mirrored)]
        img_gauss = np.dstack((r_gauss, g_gauss, b_gauss))
        list_img_a_gauss.append(img_gauss)
        list_img_a_gauss_fourier.append(make_fourier_transform(img_gauss))
        img_cycle = img_gauss

    return (list_img_a_gauss, list_img_a_gauss_fourier)


def save_gaussian_pyramid(list_gauss):
    for i, image in enumerate(list_gauss):
        imsave(f'a-gauss-{i + 1}.png', image)


def save_gaussian_pyramid_fourier(list_gauss_fourier):
    for i, image in enumerate(list_gauss_fourier):
        imsave(f'a-gauss-fourier-{i + 1}.png', image)


def create_laplacian_pyramid(img, sigma, n_gauss_layers):
    list_img_gauss, _ = create_gaussian_pyramid(
        img, sigma, n_gauss_layers)

    list_img_laplace = []
    list_img_laplace_fourier = []
    for i in range(len(list_img_gauss) - 1):
        img_laplace = img_as_ubyte(img_as_float(list_img_gauss[i]) -
                                   img_as_float(list_img_gauss[i + 1]))
        img_laplace_fourier = make_fourier_transform(img_laplace)
        list_img_laplace.append(img_laplace)
        list_img_laplace_fourier.append(img_laplace_fourier)

    list_img_laplace.append(list_img_gauss[n_gauss_layers - 1])

    return (list_img_laplace, list_img_laplace_fourier)


def save_laplacian_pyramid(list_laplace):
    for i, image in enumerate(list_laplace):
        imsave(f'a-laplace-{i + 1}.png', image)


def save_laplacian_pyramid_fourier(list_laplace_fourier):
    for i, image in enumerate(list_laplace_fourier):
        imsave(f'a-laplace-fourier-{i + 1}.png', image)


def binarize_img(img):
    return img_as_ubyte((img > 128).astype('float'))


def blend_images(img_1, img_2, mask, sigma, n_gauss_layers):
    list_mask_gauss, _ = create_gaussian_pyramid(
        mask, sigma, n_gauss_layers + 1)
    list_img_1_laplace, _ = create_laplacian_pyramid(
        img_1, sigma, n_gauss_layers)
    list_img_2_laplace, _ = create_laplacian_pyramid(
        img_2, sigma, n_gauss_layers)

    list_blended = []
    for i in range(n_gauss_layers):
        layer_blended = img_as_float(list_mask_gauss[i + 1]) * img_as_float(list_img_1_laplace[i]) + \
            (1 - img_as_float(list_mask_gauss[i + 1])) * \
            img_as_float(list_img_2_laplace[i])
        list_blended.append(layer_blended)

    img_blended = img_as_ubyte(np.clip(sum(list_blended), 0, 1))

    return (list_blended, img_blended)


def create_mask(shape_0, shape_1, shape_2):
    mask = np.zeros((shape_0, shape_1, shape_2))
    mask[:, :mask.shape[1] // 2, :] = 1.0
    mask = img_as_ubyte(mask)
    return mask


if __name__ == "__main__":

    sigma = 2.3
    n_gauss_layers = 7

    # img_a = imread('a.png')
    img_a = imread('apple.png')[..., :3]

    # print_img_params(img_a)

    # list_img_a_gauss, list_img_a_gauss_fourier = create_gaussian_pyramid(
    #     img_a, sigma, n_gauss_layers)

    # save_gaussian_pyramid(list_img_a_gauss)
    # save_gaussian_pyramid_fourier(list_img_a_gauss_fourier)

    # print(len(list_img_a_gauss))  # = n_gauss_layers
    # print(len(list_img_a_gauss_fourier))  # = n_gauss_layers

    # print(create_kernel.__doc__)

    # list_img_a_laplace, list_img_a_laplace_fourier = create_laplacian_pyramid(
    #     img_a, sigma, n_gauss_layers)

    # save_laplacian_pyramid(list_img_a_laplace)
    # save_laplacian_pyramid_fourier(list_img_a_laplace_fourier)

    # img_b = imread('b.png')
    img_b = imread('orange.png')[..., :3]

    # print_img_params(img_b)

    # mask = binarize_img(imread('mask.png'))
    mask = create_mask(img_a.shape[0], img_a.shape[1], img_a.shape[2])

    # print_img_params(mask)

    list_blended, img_blended = blend_images(
        img_b, img_a, mask, sigma, n_gauss_layers)

    # print_img_params(img_blended)

    imshow(img_blended)
    # imsave(f'orange-apple-blended-{sigma}-{n_gauss_layers}.png', img_blended)
    plt.show()
