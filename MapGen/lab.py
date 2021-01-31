"""
Created on Jan 25 2018
Last edited on May 09 2018
@author: M.Sc. Dennis Wittich
"""

from numba import jit, float64, int64, types, bool_
import imageio
import numpy as np
import PIL.Image
import IPython.display


# ========================= PUBLIC METHODS ======================================

@jit(nopython=True)
def extend_same(I, n):
    """Extends an image by 'n' pixels by repeating the image border.

        Parameters
        ----------
        I : ndarray of float64
            3D array representation of an image
        n : int
            Width of the border [n > 0]

        Returns
        -------
        out : ndarray of float64
            Extended image

        Notes
        -----
        Function accepts single and multi channel images.
    """

    h, w, d = I.shape
    s = n

    new_shape = (h + 2 * s, w + 2 * s, d)
    Iext = np.zeros(new_shape, dtype=np.float64)

    Iext[s:h + s, s:w + s, :] = np.copy(I)

    Iext[:s, :s, :] = np.ones((s, s, d)) * I[0, 0, :]
    Iext[:s, s + w:, :] = np.ones((s, s, d)) * I[0, -1, :]
    Iext[s + h:, :s, :] = np.ones((s, s, d)) * I[-1, 0, :]
    Iext[s + h:, s + w:, :] = np.ones((s, s, d)) * I[-1, -1, :]

    for x in range(h):
        target_row = s + x
        value_left = I[x, 0, :]
        value_right = I[x, -1, :]
        for y in range(s):
            Iext[target_row, y, :] = value_left
            Iext[target_row, y - s, :] = value_right

    for y in range(w):
        target_column = s + y
        value_up = I[0, y, :]
        value_low = I[-1, y, :]
        for x in range(s):
            Iext[x, target_column, :] = value_up
            Iext[x - s, target_column, :] = value_low

    return Iext


# ========================= PRIVATE METHODS ====================================

@jit(nopython=True)
def __sw_convolution(img, filter):
    fs = filter.shape[0]
    hfs = int(fs / 2)
    h, w, d = img.shape

    out_image = np.zeros((h - 2 * hfs, w - 2 * hfs, d), dtype=np.float64)

    for x in range(0, h - 2 * hfs):
        for y in range(0, w - 2 * hfs):
            for z in range(d):
                v = 0.0
                for x_ in range(fs):
                    for y_ in range(fs):
                        v += img[x + x_, y + y_, z] * filter[x_, y_]

                out_image[x, y, z] = v

    return out_image


@jit(nopython=True)
def __fast_sw_convolution_sv(img, u0, v0):
    fs = u0.shape[0]
    hfs = int(fs / 2)

    h, w, d = img.shape

    mid_image = np.zeros((h - 2 * hfs, w - 2 * hfs, d), dtype=np.float64)
    for x in range(0, h - 2 * hfs):
        for y in range(0, w - 2 * hfs):
            for z in range(d):
                v = 0.0
                for x_ in range(fs):
                    v += img[x + x_, y + hfs, z] * u0[x_]
                mid_image[x, y, z] = v

    mid_image = extend_same(mid_image, hfs)

    out_image = np.zeros((h - 2 * hfs, w - 2 * hfs, d), dtype=np.float64)

    for x in range(0, h - 2 * hfs):
        for y in range(0, w - 2 * hfs):
            for z in range(d):
                v = 0.0
                for y_ in range(fs):
                    v += mid_image[x + hfs, y + y_, z] * v0[y_]
                out_image[x, y, z] = v

    return out_image


@jit(nopython=True)
def __extend_map_2D(I, n):
    h, w = I.shape
    s = n

    new_shape = (h + 2 * s, w + 2 * s)
    out_image = np.zeros(new_shape, dtype=np.int64)

    out_image[s:h + s, s:w + s] = np.copy(I)

    out_image[:s, :s] = np.ones((s, s)) * I[0, 0]
    out_image[:s, s + w:] = np.ones((s, s)) * I[0, -1]
    out_image[s + h:, :s] = np.ones((s, s)) * I[-1, 0]
    out_image[s + h:, s + w:] = np.ones((s, s)) * I[-1, -1]

    for x in range(h):
        target_row = s + x
        value_left = I[x, 0]
        value_right = I[x, -1]
        for y in range(s):
            out_image[target_row, y] = value_left
            out_image[target_row, y - s] = value_right

    for y in range(w):
        target_column = s + y
        value_up = I[0, y]
        value_low = I[-1, y]
        for x in range(s):
            out_image[x, target_column] = value_up
            out_image[x - s, target_column] = value_low

    return out_image


@jit(nopython=True)
def __get_min_coords_2d_threshold(matrix, t):
    h, w = matrix.shape

    min_coords = np.zeros((2), dtype=np.int64)

    for x in range(h):
        for y in range(w):
            val = matrix[x, y]
            if val < t:
                t = val
                min_coords[:] = (x, y)

    return (min_coords, t)


# ========================= PUBLIC METHODS ======================================


@jit(nopython=True)
def add_mixed_noise(I, s_white, p_salt_pepper):
    """Adds white noise and salt-pepper noise to an image.

        Parameters
        ----------
        I : ndarray of float64
            3D array representation of an image
        s_white : float
            standard deviation of white noise [s >= 0.0]
        p_salt_pepper : float
            probability to set a pixel to black or white [0.0 >= p >= 1.0]

        Returns
        -------
        out : ndarray of float64
            Copy of the input image with noise

        Notes
        -----
        Function adds white noise to the left half of the image and salt-pepper noise to the right half.
        Function accepts single or multi channel images.
    """

    assert s_white >= 0.0, "Sigma of white noise must not be negative!"
    assert p_salt_pepper >= 0.0 and p_salt_pepper <= 1.0, "Prob. of salt-pepper noise must be between 0 and 1!"

    h, w, d = I.shape
    J = I.copy()

    for x in range(h):
        for y in range(w):
            if y < w / 2:
                for z in range(d):
                    v = J[x, y, z] + np.random.randn(1)[0] * s_white
                    v = max(min(255.0, v), 0.0)
                    J[x, y, z] = v
            elif y == w / 2:
                J[x, y, :] = 0.0
            else:
                if np.random.random() < p_salt_pepper:
                    if np.random.random() < (p_salt_pepper / 2):
                        J[x, y, :] = 255.0
                    else:
                        J[x, y, :] = 0.0
    return J


@jit(nopython=True)
def get_valid_neighbours_dist(h, w, x, y, d):
    """Computes valid neighbour coordinates of a pixel within distance 'd'.

        Parameters
        ----------
        h : int
            height of the image
        w : int
            width of the image
        x : int
            x coordinate of pixel
        y : int
            y coordinate of pixel
        d : int
            maximum manhattan distance neighbour - pixel

        Returns
        -------
        out : list of (int, int)-tuples
            List of coordinates of valid neighbours as tuples

        Notes
        -----
        Distance 'd' denotes the maximum manhattan distance between the pixel and the neighbours.
        E.g. 'd' = 1 will return the 8 direct neighbours, 'd' = 2 will return 14 neighbours.
        Valid means no neighbours will be in the list, whose coordinates exceed the image boundaries.
        Function accepts single or multi channel images but will always return 2D coordinates in image space.
    """

    assert h > 0 and w > 0 and d > 0, "h, w and d must not be greater zero!"

    neighbours = [(0, 0)][1:]

    for x_ in range(max(0, x - d), min(h, x + 1 + d)):
        for y_ in range(max(0, y - d), min(w, y + 1 + d)):
            if x_ != x or y_ != y:
                neighbours += [(x_, y_)]

    return neighbours


@jit(nopython=True)
def get_valid_neighbours(h, w, x, y):
    """Computes valid neighbour coordinates of a pixel within distance 1.

        Parameters
        ----------
        h : int
            height of the image
        w : int
            width of the image
        x : int
            x coordinate of pixel
        y : int
            y coordinate of pixel

        Returns
        -------
        out : list of of (int, int)-tuples
            List of coordinates of valid neighbours as tuples

        Notes
        -----
        Same as get_valid_neighbours_dist with param 'd' = 1.
    """
    return get_valid_neighbours_dist(h, w, x, y, 1)


@jit(nopython=True)
def get_connected_components(I):
    """Computes a segment map from connected components (by value).

        Parameters
        ----------
        I : ndarray of float64
            3D array representation of an image

        Returns
        -------
        out : ndarray of int64
            2D array where each entry corresponds to the component ID

        Notes
        -----
        Function searches for connected regions in an image with the same values!
        Function accepts single or multi channel images. Segment IDs will start with zero!
    """

    h, w, d = I.shape

    ### MAP COORDINATES TO HORIZONTAL LABEL
    label_map_horizontal = np.zeros((h, w), dtype=np.int64)
    next_label = 1
    for x in range(h):
        for y in range(w):
            merge_h = y > 0
            if merge_h:
                for z in range(d):
                    if I[x, y - 1, z] != I[x, y, z]:
                        merge_h = False
                        break
            if merge_h:
                label_map_horizontal[x, y] = label_map_horizontal[x, y - 1]
            else:
                merge_v = x > 0
                if merge_v:
                    for z in range(d):
                        if I[x - 1, y, z] != I[x, y, z]:
                            merge_v = False
                            break
                if merge_v:
                    label_map_horizontal[x, y] = label_map_horizontal[x - 1, y]
                else:
                    label_map_horizontal[x, y] = next_label
                    next_label += 1

    ### MAP HORIZONTAL INDICES TO FINAL INDICES
    final_labels_length = next_label
    final_labels = np.ones((final_labels_length), dtype=np.int64) * -1
    next_final_label = 0
    for x in range(h):
        for y in range(w):
            try_merge = x > 0
            h_label = label_map_horizontal[x, y]
            merge = try_merge
            if try_merge:
                upper_h_label = label_map_horizontal[x - 1, y]
                if h_label == upper_h_label or final_labels[h_label] == final_labels[upper_h_label]:
                    merge = False
                else:
                    for z in range(d):
                        if I[x - 1, y, z] != I[x, y, z]:
                            merge = False
                            break
                if merge:
                    if final_labels[h_label] == -1:
                        final_labels[h_label] = final_labels[upper_h_label]
                    elif final_labels[upper_h_label] != final_labels[h_label]:
                        to_change_final = final_labels[h_label]
                        for i in range(final_labels_length):
                            if final_labels[i] == to_change_final:
                                final_labels[i] = final_labels[upper_h_label]
            if not merge and final_labels[h_label] == -1:
                final_labels[h_label] = next_final_label
                next_final_label += 1

    ### MAP FINAL INDICES TO INDICES WITHOUT GAPS (STARTING FROM ZERO)
    no_gap_labels_length = next_final_label
    no_gap_labels = np.ones((no_gap_labels_length), dtype=np.int64) * -1
    no_gap_index = 0
    for x in range(h):
        for y in range(w):
            if no_gap_labels[final_labels[label_map_horizontal[x, y]]] == -1:
                no_gap_labels[final_labels[label_map_horizontal[x, y]]] = no_gap_index
                no_gap_index += 1

    label_map = np.zeros((h, w), dtype=np.int64)
    for x in range(h):
        for y in range(w):
            label_map[x, y] = no_gap_labels[final_labels[label_map_horizontal[x, y]]]

    return label_map


@jit(nopython=True)
def get_contour_image(S):
    """Computes a contour ID map from a segment map.

        Parameters
        ----------
        S : ndarray of int64
            2D array representing the segment map

        Returns
        -------
        out : ndarray of int64
            2D array representing the contour map

        Notes
        -----
        Each entry of the contour map corresponds to the component ID,
        if the entry it is on the contour of that component, else -1.
        The contour map is computed via contour tracing.
    """

    h, w = S.shape
    num_s = np.max(S) + 1
    C = np.ones((h, w), dtype=np.int64) * -1

    for s in range(num_s):

        # GET CONTOUR SEED FOR CLASS 'c'
        start_pixel = (-1, -1)
        for x in range(h):
            for y in range(w):
                if S[x, y] == s:
                    start_pixel = (x, y)
                    break
            if start_pixel[0] >= 0:
                break

        if start_pixel == (-1, -1):
            continue

        # FOLLOW CONTOUR OF SEGMENT 's'
        search_dir = 0
        current_pixel = start_pixel
        done = False
        while True:
            cpx, cpy = current_pixel
            C[cpx, cpy] = s
            for i in range(8):  # while(True) failes with jit..
                if search_dir == 0 and cpx > 0 and S[cpx - 1, cpy] == s:  # start at pixel above
                    current_pixel = (cpx - 1, cpy)
                    break
                elif search_dir == 45 and cpx > 0 and cpy < w - 1 and S[cpx - 1, cpy + 1] == s:
                    current_pixel = (cpx - 1, cpy + 1)
                    break
                elif search_dir == 90 and cpy < w - 1 and S[cpx, cpy + 1] == s:
                    current_pixel = (cpx, cpy + 1)
                    break
                elif search_dir == 135 and cpx < h - 1 and cpy < w - 1 and S[cpx + 1, cpy + 1] == s:
                    current_pixel = (cpx + 1, cpy + 1)
                    break
                elif search_dir == 180 and cpx < h - 1 and S[cpx + 1, cpy] == s:
                    current_pixel = (cpx + 1, cpy)
                    break
                elif search_dir == 225 and cpx < h - 1 and cpy > 0 and S[cpx + 1, cpy - 1] == s:
                    current_pixel = (cpx + 1, cpy - 1)
                    break
                elif search_dir == 270 and cpy > 0 and S[cpx, cpy - 1] == s:
                    current_pixel = (cpx, cpy - 1)
                    break
                elif search_dir == 315 and cpx > 0 and cpy > 0 and S[cpx - 1, cpy - 1] == s:
                    current_pixel = (cpx - 1, cpy - 1)
                    break
                search_dir = (search_dir + 45) % 360
                if search_dir == 0 and start_pixel == current_pixel:
                    done = True
                    break
            search_dir = (search_dir + 270) % 360

            if done:
                break
    return C


@jit(nopython=True)
def segments_to_image(S):
    """Computes an segment image from a segment map.

        Parameters
        ----------
        S : ndarray of int64
            2D array representing the segment map

        Returns
        -------
        out : ndarray of float64
            3D array representing the segment image

        Notes
        -----
        Pixels with the segment ID -1 will be black.
        Other IDs will be mapped to a random color with fixed seed.
    """

    h, w = S.shape
    class_image = np.zeros((h, w, 3), dtype=np.float64)
    num_classes = np.max(S)
    np.random.seed(42)
    colors = np.random.rand(num_classes + 1, 3) * 200 + 55
    colors[0, :] = 255
    for x in range(h):
        for y in range(w):
            if S[x, y] >= 0:
                class_image[x, y, :] = colors[S[x, y], :]
    return class_image


@jit(nopython=True)
def convolution(I, H):
    """Convolves an image with a filter matrix. Returns the filtered image.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image to convolve
        H : ndarray of float64
            2D array, filter to convolve with

        Returns
        -------
        out : ndarray of float64
            3D array, convolved image

        Notes
        -----
        Input array will be extended with border values,
        so the resulting array will have the same shape as the input array.
        Function accepts single or multi channel images.
        Convolution will be performed channel-wise, so the filtered image
        will have the same number of channels as the input image.
    """

    assert I.ndim == 3, "Image to convolve must have three dimensions."
    fs = H.shape[0]
    assert fs % 2 != 0, "Filter size must be odd!"
    assert H.shape[0] == H.shape[1], "Function only supports square filter matrices."
    hfs = int(fs / 2)
    image = extend_same(I, hfs)
    u, s, vh = np.linalg.svd(H, True)
    if s[1] < 1e-15:
        s0_root = np.sqrt(s[0])
        u0 = u[:, 0] * s0_root
        v0 = vh[0, :] * s0_root
        return __fast_sw_convolution_sv(image, u0, v0)
    if s[2] < 1e-15:
        s0_root = np.sqrt(s[0])
        u0 = u[:, 0] * s0_root
        v0 = vh[0, :] * s0_root
        s1_root = np.sqrt(s[1])
        u1 = u[:, 1] * s1_root
        v1 = vh[1, :] * s1_root
        return __fast_sw_convolution_sv(image, u0, v0) + \
               __fast_sw_convolution_sv(image, u1, v1)
    else:
        return __sw_convolution(image, H)


@jit(nopython=True)
def normalize(I):
    """Normalizes a single channel image to the range 0.0 - 255.0.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image to normalize

        Returns
        -------
        out : ndarray of float64
            3D array, normalized image

        Notes
        -----
        If minimum and maximum value in 'I' are identical, a copy of 'I' is returned.
    """

    min_value = np.min(I)
    max_value = np.max(I)

    if max_value == min_value:
        return np.copy(I)

    return (I - min_value) * 255 / (max_value - min_value)  # creates a copy


@jit(nopython=True)
def gaussian_blur(I, sigma):
    """Convolves an image with a gaussian filter. Returns the filtered image.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image to convolve
        sigma : float64
            Standard deviation of the gaussian filter

        Returns
        -------
        out : ndarray of float64
            3D array, convolved image

        Notes
        -----
        Input array will be extended with border values,
        so the resulting array will have the same shape as the input array.
        Function accepts single and multi channel images.
    """

    # code cryptified (Exercise in Lab)

    q = int(6 * sigma)
    t = sigma ** 2 * 2
    q += (1 if q % 2 == 0 else 0)
    f = np.zeros((q, q), dtype=np.float64)
    o = q // 2
    k = o + 1
    for a in range(k):
        for b in range(a, k):
            g = 1.0 / (np.pi * t) * np.exp(-((a - o) ** 2 + (b - o) ** 2) / t)
            f[a, b] = f[-a - 1, b] = f[-a - 1, -b - 1] = f[a, -b - 1] = g
            if a != b:
                f[b, a] = f[-b - 1, a] = f[-b - 1, -a - 1] = f[b, -a - 1] = g

    return convolution(I, f)


@jit(nopython=True)
def gaussian_derivatives(I, sigma):
    """Convolves an image with the derivative of gaussian filters. Returns the derivatives in x and y direction.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image to convolve
        sigma : float64
            Standard deviation of gaussian

        Returns
        -------
        out : ndarray of float64, ndarray of float64
            derivative in x direction, derivative in y direction

        Notes
        -----
        Input array will be extended with border values,
        so the resulting arrays will have the same shape as the input array.\n
    """

    # code cryptified (Exercise in Lab)

    q = int(6 * sigma)
    t = sigma ** 2 * 2
    p = - 2 * np.pi * sigma ** 4
    q += (1 if q % 2 == 0 else 0)
    f = np.zeros((q, q), dtype=np.float64)
    f2 = np.zeros_like(f)
    o = q // 2
    for a in range(q):
        for b in range(q):
            j = np.exp(-((a - o) ** 2 + (b - o) ** 2) / t)
            g = (a - o) / p * j
            g2 = (b - o) / p * j
            f[a, b] = g
            f2[a, b] = g2

    return convolution(I, f), convolution(I, f2)


@jit(nopython=True)
def gaussian_gradients(I, sigma):
    """Computes the gradients of an image. Returns magnitudes and directions.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing an image
        sigma : float64
            Standard deviation of gaussian derivatives

        Returns
        -------
        out : ndarray of float64, ndarray of float64
            gradient magnitudes, gradient directions in radiants

        Notes
        -----
        Input array will be extended with border values,
        so the resulting arrays will have the same shape as the input array.\n
    """

    dx, dy = gaussian_derivatives(I, sigma)
    return np.sqrt(np.square(dx) + np.square(dy)), np.arctan2(dy, dx)


@jit(nopython=True)
def prettify_gradients(magnitudes, angles):
    """Computes a visualization of the gradients.

        Parameters
        ----------
        magnitudes : ndarray of float64
            3D array representing magnitudes
        angles : ndarray of float64
            3D array representing angles

        Returns
        -------
        out : ndarray of float64
            visualization of gradients as rgb image

        Notes
        -----
        Function maps angles to hue and magnitudes to saturation.
        Gradient angles must be given in rad.
    """

    h_, w_, d_ = angles.shape
    assert d_ == 1, 'input arrays must have a depth of one!'
    assert angles.shape == magnitudes.shape, 'shapes of arrays must match!'

    magnitudes = magnitudes / np.max(magnitudes)
    rgb_image = np.zeros((h_, w_, 3), dtype=np.float64)
    for x in range(h_):
        for y in range(w_):
            h, s, v = angles[x, y, 0], magnitudes[x, y, 0], 1
            h = h / np.pi * 180
            while h < 0:
                h += 360
            while h >= 360:
                h -= 360
            if h == 0.0:
                h += 0.000000001
            h_i = int(h // 60) % 6
            f = h / 60 - h_i
            p = v * (1 - s)
            q = v * (1 - s * f)
            t = v * (1 - s * (1 - f))
            if h_i == 0:
                rgb = (v, t, p)
            elif h_i == 1:
                rgb = (q, v, p)
            elif h_i == 2:
                rgb = (p, v, t)
            elif h_i == 3:
                rgb = (p, q, v)
            elif h_i == 4:
                rgb = (t, p, v)
            else:
                rgb = (v, p, q)
            rgb_image[x, y, :] = rgb

    rgb_image *= 255
    return rgb_image


@jit(nopython=True)
def rgb2hsv(I):
    """Converts a image from RGB to HSV space.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the RGB image in range 0.0 - 255.0

        Returns
        -------
        out : ndarray of float64
            3D array, image in HSV space

        Notes
        -----
        Hue values will be in degree [0-360].
        Saturations will be in range [0-1].
        Values will be in range [0-1].
    """

    h, w, d = I.shape
    assert d == 3, "Only supported for color images."
    I = np.copy(I) / 255

    hsv_image = np.zeros((h, w, 3), dtype=np.float64)
    for x in range(h):
        for y in range(w):
            r, g, b = I[x, y, :]
            v_max = np.max(I[x, y, :])
            v_min = np.min(I[x, y, :])

            # HUE 0 - 360
            hue = 0.0
            if v_max > v_min:
                if r == v_max:
                    hue = 60 * (g - b) / (v_max - v_min)
                elif g == v_max:
                    hue = 120 + 60 * (b - r) / (v_max - v_min)
                elif b == v_max:
                    hue = 240 + 60 * (r - g) / (v_max - v_min)
                if hue < 0:
                    hue += 360

            # SATURATION 0 - 1
            sat = 0.0
            if v_max > 0.0:
                sat = (v_max - v_min) / v_max

            # VALUE 0 - 1
            val = v_max
            hsv_image[x, y, :] = (hue, sat, val)
    return hsv_image


@jit(nopython=True)
def draw_best_lines(I, hough_space, num_lines, threshold):
    """Computes and draws the n best lines to the image and marks them in the hough space.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image
        hough_space : ndarray of float64
            2D array representing the hough space (x-axis = angle, y-axis = distance)
        num_lines : int
            the number of best lines to draw [ >= 0]
        threshold : float
            minimum value in hough space. Saves computation time while searching for maximas [ >= 0.0]

        Returns
        -------
        out : ndarray of float64, ndarray of float64
            tuple of 3D arrays, image with best lines, hough space with maximas (as image)

        Notes
        -----
        Function determines n best lines and draws them to the image and hough space.
    """

    hough_space = np.expand_dims(hough_space, 2)
    hs_with_maxs = np.dstack((hough_space, hough_space, hough_space)) / np.max(hough_space) * 255
    hs_blurred = gaussian_blur(hough_space, 1.0)[:, :, 0]

    hh, hw = hs_blurred.shape
    max_dist = hw // 2

    # Step 1: Compute local maximas

    local_maxima = [(0, 0)][1:]
    for x in range(hh):
        for y in range(hw):
            v = hs_blurred[x, y]
            if (v < threshold):
                continue
            N = get_valid_neighbours_dist(hh, hw, x, y, 7)
            is_local_max = True
            for nx, ny in N:
                if hs_blurred[nx, ny] > v:
                    is_local_max = False
                    break
            if is_local_max:
                local_maxima += [(x, y)]

    # Step 2: Get 'num_lines' best lines (and draw to hough space)

    lines = [(0, 0)][1:]  # angle, distance
    num_lines = min(num_lines, len(local_maxima))
    while len(lines) < num_lines:
        best_line = (0, 0)
        best_value = 0.0
        for a, d in local_maxima:
            if (a, d) in lines:
                continue
            if hs_blurred[a, d] > best_value:
                best_value = hs_blurred[a, d]
                best_line = (a, d)
        hs_with_maxs[best_line[0], best_line[1], :] = [255, 0, 0]
        S = get_valid_neighbours_dist(hh, hw, best_line[0], best_line[1], 3)
        for sx, sy in S:
            hs_with_maxs[sx, sy, :] = (255, 0, 0)
        lines += [best_line]

    # Step 3: Compute normal vector for each line

    lines_normal = [(0.0, 0.0, 0.0)][1:]  # nx, ny, distance
    for line in lines:
        rad = line[0] / 180 * np.pi
        dis = float(line[1]) - max_dist
        lines_normal += [(np.cos(rad), np.sin(rad), dis)]

    # Step 4: Draw lines

    h, w, _ = I.shape
    img_with_lines = np.concatenate((I, I, I), 2)
    for x in range(h):
        for y in range(w):
            for (normal_x, normal_y, dist) in lines_normal:
                if abs(x * normal_x + y * normal_y - dist) < 1.5:
                    img_with_lines[x, y, :] = (255, 0, 0)

    return img_with_lines, hs_with_maxs


@jit(nopython=True)
def region_growing_rgb(I, max_dist, min_pixels=1, smoothing=0):
    """Performs region growing on a RGB image.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image
        max_dist : float
            maximum euclidean rgb distance to add a neighbour to a region [ >= 0.0]
        min_pixels : int
            minimum size of a region [ > 0]
        smoothing : int
            size of smoothing distance in pixels [ >= 0]

        Returns
        -------
        out : ndarray of int64
            region map IDs continuous starting from 0

        Notes
        -----
        When smoothing is applied ( > 0), the minimum pixel size is no longer guaranteed!
    """

    max_dist_sq = max_dist ** 2
    h, w, d = I.shape
    assert d == 3, "Only color images supported!"

    S = np.ones((h, w), dtype=np.int64) * -1
    seeds = [(0, 0)][1:]
    for x in range(h):
        for y in range(w):
            seeds.append((x, y))
    current_seg_id = -1
    while len(seeds) > 0:
        sx, sy = seeds.pop()
        if S[sx, sy] >= 0:
            continue
        current_seg_id += 1
        S[sx, sy] = current_seg_id
        members = [(sx, sy)]
        sum_colors = np.copy(I[sx, sy])
        num_pixels = 1
        nearest_segment_distance = -1
        nearest_seg_id = -1
        mean_colors = np.copy(sum_colors)  / num_pixels

        ### GROWING

        added_members = True
        while added_members:
            added_members = False
            for mx, my in members:
                neighbours = get_valid_neighbours(h, w, mx, my)
                for nx, ny in neighbours:
                    ncol = I[nx, ny]
                    ns = S[nx, ny]
                    g_dist = np.sum(np.square(ncol - mean_colors))
                    if ns != current_seg_id and ns >= 0 and (
                            g_dist < nearest_segment_distance or nearest_seg_id == -1):
                        nearest_segment_distance = g_dist
                        nearest_seg_id = ns

                    if ns < 0 and g_dist < max_dist_sq:
                        S[nx, ny] = current_seg_id
                        num_pixels += 1
                        sum_colors += ncol
                        mean_colors = sum_colors / num_pixels
                        members.append((nx, ny))
                        added_members = True

        ### MIN SIZE TEST

        if num_pixels < min_pixels:
            for x in range(h):
                for y in range(w):
                    if S[x, y] == current_seg_id:
                        S[x, y] = nearest_seg_id
            if nearest_seg_id == -1:
                seeds.reverse()
                seeds.append((x, y))
                seeds.reverse()
            current_seg_id -= 1

    ### SMOOTHING
    if smoothing > 0:
        n = smoothing
        if n % 2 == 0:
            n += 1
        R = np.zeros((h, w, 1), dtype=np.float64)
        hfs = n // 2
        S_ext = __extend_map_2D(S, hfs)
        h, w = S_ext.shape
        for x in range(0, h - 2 * hfs):
            for y in range(0, w - 2 * hfs):
                area = S_ext[x:x + n, y:y + n]
                R[x, y] = np.argmax(np.bincount(area.flatten()))
        S = get_connected_components(R)

    return S


@jit(nopython=True)
def region_growing_hsv(I, max_dist, min_pixels=1, smoothing=0):
    """Performs region growing on an color image in HSV space.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image
        max_dist : float
            maximum hsv distance to add a neighbour to a region [ >= 0.0]
        min_pixels : int
            minimum size of a region [ > 0]
        smoothing : int
            size of smoothing distance in pixels [ >= 0]

        Returns
        -------
        out : ndarray of int64
            region map IDs continuous starting from 0

        Notes
        -----
        The hsv distance is a experimental metric which weights the hue distance,
        depending on the mean saturation and value of a pixel and region mean.
        When smoothing is applied ( > 0), the minimum pixel size is no longer guaranteed!
    """

    max_dist_sq = max_dist ** 2
    h, w, d = I.shape
    assert d == 3, "Only color images supported!"
    HSV = rgb2hsv(I)

    S = np.ones((h, w), dtype=np.int64) * -1
    seeds = [(0, 0)][1:]
    for x in range(h):
        for y in range(w):
            seeds.append((x, y))
    current_seg_id = -1
    while len(seeds) > 0:
        sx, sy = seeds.pop()
        if S[sx, sy] >= 0:
            continue
        current_seg_id += 1
        S[sx, sy] = current_seg_id
        members = [(sx, sy)]
        sum_colors = np.copy(HSV[sx, sy])
        num_pixels = 1
        nearest_segment_distance = -1
        nearest_seg_id = -1
        mean_colors = np.copy(sum_colors)  # / num_pixels

        ### GROWING

        added_members = True
        while added_members:
            added_members = False
            for mx, my in members:
                neighbours = get_valid_neighbours(h, w, mx, my)
                for nx, ny in neighbours:
                    ncol = HSV[nx, ny]
                    ns = S[nx, ny]

                    dist = ncol - mean_colors
                    if dist[0] > 180:
                        dist[0] -= 360
                    elif dist[0] < -180:
                        dist[0] += 360

                    dist[0] *= (ncol[1] + mean_colors[1] + ncol[2] + mean_colors[2]) / (4 * 180)

                    g_dist = np.sum(np.square(dist))
                    if ns != current_seg_id and ns >= 0 and (
                            g_dist < nearest_segment_distance or nearest_seg_id == -1):
                        nearest_segment_distance = g_dist
                        nearest_seg_id = ns

                    if ns < 0 and g_dist < max_dist_sq:
                        S[nx, ny] = current_seg_id
                        num_pixels += 1

                        sum_colors += ncol
                        mean_colors = sum_colors / num_pixels
                        if mean_colors[0] < 0:
                            mean_colors[0] += 360
                        elif mean_colors[0] > 360:
                            mean_colors[0] -= 360
                        members.append((nx, ny))
                        added_members = True

        ### MIN SIZE TEST

        if num_pixels < min_pixels:
            for x in range(h):
                for y in range(w):
                    if S[x, y] == current_seg_id:
                        S[x, y] = nearest_seg_id
            if nearest_seg_id == -1:
                seeds.reverse()
                seeds.append((x, y))
                seeds.reverse()
            current_seg_id -= 1

    # ====== SMOOTHING =========

    if smoothing > 0:
        n = smoothing
        if n % 2 == 0:
            n += 1
        R = np.zeros((h, w, 1), dtype=np.float64)
        hfs = n // 2
        S_ext = __extend_map_2D(S, hfs)
        h, w = S_ext.shape
        for x in range(0, h - 2 * hfs):
            for y in range(0, w - 2 * hfs):
                area = S_ext[x:x + n, y:y + n]
                R[x, y] = np.argmax(np.bincount(area.flatten()))
        S = get_connected_components(R)

    return S


@jit(nopython=True)
def watershed_transform(I, sigma, seed_threshold):
    """Performs a watershed transformation on an image.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image
        sigma : float
            standard deviation of gaussion to compute gradients
        seed_threshold : float
            magnitude threshold to start a new seed

        Returns
        -------
        out : ndarray of int64
            region map IDs continuous starting from 0

        Notes
        -----
        Implemented with Meyer's flooding algorithm.
    """
    h, w, d = I.shape
    if d > 1:
        I = (I[:, :, 0:1:] + I[:, :, 1:2:] + I[:, :, 2:3:]) / 3.0

    amplitude_map = gaussian_gradients(I, sigma)[0][:,:,0]
    max_amp_plus = np.max(amplitude_map) + 1

    (gmx, gmy), gmv = __get_min_coords_2d_threshold(amplitude_map, max_amp_plus)
    num_candidates = 1

    candidate_map = np.ones((h, w), dtype=np.float64) * max_amp_plus
    visited_map = np.zeros((h, w), dtype=np.bool_)
    label_map = np.zeros((h, w), dtype=np.int64)

    next_label = 1

    candidate_map[gmx, gmy] = gmv
    label_map[gmx, gmy] = next_label
    amplitude_map[gmx, gmy] = max_amp_plus

    # second best minimum
    (gmx, gmy), gmv = __get_min_coords_2d_threshold(amplitude_map, max_amp_plus)

    while num_candidates > 0:
        (cx, cy), cv = __get_min_coords_2d_threshold(candidate_map, max_amp_plus)

        # check for new seeds
        if cv - seed_threshold > gmv:
            if not (candidate_map[gmx, gmy] < max_amp_plus or visited_map[gmx, gmy]):
                candidate_map[gmx, gmy] = gmv
                num_candidates += 1
                next_label += 1
                label_map[gmx, gmy] = next_label
                cx = gmx
                cy = gmy

            (gmx, gmy), gmv = __get_min_coords_2d_threshold(amplitude_map, max_amp_plus)

        # remove candidate from candidates and add to visited
        amplitude_map[cx, cy] = max_amp_plus
        candidate_map[cx, cy] = max_amp_plus
        visited_map[cx, cy] = True
        num_candidates -= 1

        neighbours = get_valid_neighbours(h, w, cx, cy)
        # num_neighbours = len(neighbours)

        can_be_labeled = True
        label_vote = 0
        for nx, ny in neighbours:
            # nx, ny = neighbours[n, :]
            if not (candidate_map[nx, ny] < max_amp_plus or visited_map[nx, ny]):
                candidate_map[nx, ny] = amplitude_map[nx, ny]
                num_candidates += 1

            label = label_map[nx, ny]

            if label == 0:
                continue

            if label_vote == 0:
                label_vote = label
            elif not label_vote == label:
                can_be_labeled = False

        if can_be_labeled and (not label_map[cx, cy]):
            label_map[cx, cy] = label_vote

    return label_map


@jit(nopython=True)
def median_filter(I, n):
    """Applies a median filter to an image. Returns the filtered image.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image to convolve
        n : float64
            size of the median filter

        Returns
        -------
        out : ndarray of float64
            3D array, convolved image

        Notes
        -----
        Input array will be extended with border values,
        so the resulting array will have the same shape as the input array.
        Function accepts single or multi channel images.
    """

    # code cryptified (Exercise in Lab)

    if n % 2 == 0:
        n += 1
    q = np.zeros_like(I)
    s = n // 2
    k = extend_same(I, s)
    j, t, w = k.shape
    for i in range(0, j - 2 * s):
        for p in range(0, t - 2 * s):
            for x in range(w):
                m = k[i:i + n, p:p + n, x]
                q[i, p, x] = np.median(m)
    return q


@jit(nopython=True)
def get_geometric_segment_features(S):
    """Computes the geometric segment features, based on a segment map.

        Parameters
        ----------
        S : ndarray of int64
            2D array representing a segment map

        Returns
        -------
        out : tuple
            areas, compactnesses, MBRs, fill factors, elongations

        Notes
        -----
        Definitions
        areas : list of float
        compactnesses : list of float
        MBRs : list of tuples of floats : (angle, length, with, center x, center y)
        fill factors : list of float
        elongations : list of float
    """

    h, w = S.shape
    num_s = np.max(S) + 1
    C = get_contour_image(S)

    reg_moms = np.zeros((num_s, 2, 2), dtype=np.int64)
    cen_moms = np.zeros((num_s, 3, 3), dtype=np.int64)
    perimeters = np.zeros((num_s), dtype=np.int64)
    mass_centres = np.zeros((num_s, 2), dtype=np.float64)
    mbrs = np.zeros((num_s, 5), dtype=np.float64)  # phi, L, W, x_, y_

    for x in range(h):
        for y in range(w):
            s = S[x, y]
            reg_moms[s, 0, 0] += 1
            reg_moms[s, 1, 0] += x
            reg_moms[s, 0, 1] += y
            if C[x, y] >= 0:
                perimeters[C[x, y]] += 1

    areas = reg_moms[:, 0, 0]
    mass_centres[:, 0] = reg_moms[:, 1, 0] / areas
    mass_centres[:, 1] = reg_moms[:, 0, 1] / areas

    # cent_moms, neighbours

    for x in range(h):
        for y in range(w):
            s = S[x, y]
            x_ = x - mass_centres[s, 0]
            y_ = y - mass_centres[s, 1]
            cen_moms[s, 1, 1] += x_ * y_
            cen_moms[s, 2, 0] += x_ * x_
            cen_moms[s, 0, 2] += y_ * y_
            # cen_moms[s, .. ] += ..

    comps = np.square(perimeters) / (4 * np.pi * reg_moms[:, 0, 0])

    mbrs[:, 0] = 0.5 * np.arctan2(2 * cen_moms[:, 1, 1], (cen_moms[:, 2, 0] - cen_moms[:, 0, 2]))
    c_phis = np.cos(mbrs[:, 0])
    s_phis = np.sin(mbrs[:, 0])

    ab_min = np.ones((num_s, 2), dtype=np.float64) * h * w
    ab_max = np.ones((num_s, 2), dtype=np.float64) * h * w * -1

    for x in range(h):
        for y in range(w):
            s = C[x, y]
            if s >= 0:
                cp = c_phis[s]
                sp = s_phis[s]
                a = x * cp + y * sp
                b = -x * sp + y * cp

                if a > ab_max[s, 0]:
                    ab_max[s, 0] = a
                if a < ab_min[s, 0]:
                    ab_min[s, 0] = a

                if b > ab_max[s, 1]:
                    ab_max[s, 1] = b
                if b < ab_min[s, 1]:
                    ab_min[s, 1] = b

    ab_mean = (ab_max + ab_min) / 2
    mbrs[:, 3] = ab_mean[:, 0] * c_phis - ab_mean[:, 1] * s_phis
    mbrs[:, 4] = ab_mean[:, 0] * s_phis + ab_mean[:, 1] * c_phis

    mbrs[:, 1:3] = np.abs(ab_max - ab_min)

    fill_factors = areas / (mbrs[:, 1] * mbrs[:, 2])
    elongations = np.abs(np.log(np.abs(mbrs[:, 1] / mbrs[:, 2])))

    return areas, comps, mbrs, fill_factors, elongations


@jit(nopython=True)
def get_spectral_segment_features(S, I):
    """Computes the spectral features of all segments.

        Parameters
        ----------
        S : ndarray of int64
            2D array representing a segment map

        I : ndarray of float64
            3D array representing the image

        Returns
        -------
        out : tuple
            mean rgb values, means hsv values

        Notes
        -----
        Definitions
        mean rgb values : list of tuples of floats : (mean red, mean green, mean blue)
        mean hsv values : list of tuples of floats : (mean hue, mean saturation, mean value)
    """
    HSV = rgb2hsv(I)
    num_s = np.max(S) + 1

    means_rgb = np.zeros((num_s, 3), dtype=np.float64)
    means_hsv = np.zeros_like(means_rgb)

    for s in range(num_s):
        M = (S == s).astype(np.float64)
        sum_M = np.sum(M)
        means_rgb[s, :] = (
            np.sum(I[:, :, 0] * M) / sum_M, np.sum(I[:, :, 1] * M) / sum_M, np.sum(I[:, :, 2] * M) / sum_M)
        means_hsv[s, :] = (
            np.sum(HSV[:, :, 0] * M) / sum_M, np.sum(HSV[:, :, 1] * M) / sum_M, np.sum(HSV[:, :, 2] * M) / sum_M)

    return means_rgb, means_hsv


def draw_mbrs(I, mbrs):
    """Draws minimum bounding rectangles to an image.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image

        mbrs : list of tuples of float
            each tuple in list holds the parameters of a minimum bounding rectangle

        Returns
        -------
        out : ndarray of float64
            3D array, image with rectangles

        Notes
        -----
        Definitions
        MBRs : list of tuples of floats : (angle, length, with, center x, center y)
    """
    h, w, _ = I.shape
    J = np.copy(I)
    for phi, L, W, cx, cy in mbrs:
        cp = np.cos(phi)
        sp = np.sin(phi)
        for a in (-L / 2, L / 2):
            for b in np.linspace(-W / 2, W / 2, int(W)):
                x = int(a * cp - b * sp + cx)
                y = int(a * sp + b * cp + cy)
                if x >= 0 and x < h and y >= 0 and y < w:
                    J[x, y, :] = 0.0
                    J[x, y, 0] = 255

        for a in np.linspace(-L / 2, L / 2, int(L)):
            for b in (-W / 2, W / 2):
                x = int(a * cp - b * sp + cx)
                y = int(a * sp + b * cp + cy)
                if x >= 0 and x < h and y >= 0 and y < w:
                    J[x, y, :] = 0.0
                    J[x, y, 0] = 255
    return J


@jit(nopython=True)
def get_neighbouring_segments(S):
    """Draws minimum bounding rectangles to an image.

        Parameters
        ----------
        S : ndarray of int64
            2D array representing the segment map

        Returns
        -------
        out : ndarray of bool_
            2D array, entry at i,j = True if segments i and j are neighbours
    """
    h, w = S.shape
    num_s = np.max(S) + 1
    neighbour_matrix = np.zeros((num_s, num_s), dtype=np.bool_)
    for x in range(h):
        for y in range(w):
            s_id = S[x, y]
            N = get_valid_neighbours(h, w, x, y)
            for nx, ny in N:
                n_id = S[nx, ny]
                neighbour_matrix[s_id, n_id] = True
                neighbour_matrix[n_id, s_id] = True
    return neighbour_matrix


@jit(nopython=True)
def k_means(I, k):
    """Performs k-means clustering on an image.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing an image
        k : int
            number of means

        Returns
        -------
        out : ndarray of float64
            3D array, image, where each pixel is set to the value of closest mean

        Notes
        -----
        Distance is computed with L2 norm.
        Function takes single and multi channel images.
    """
    h, w, d = I.shape

    S = np.zeros((h, w), dtype=np.int64)
    k_map = np.zeros_like(S)
    k_map_ = np.copy(k_map)
    counts = np.zeros(k, dtype=np.float64)
    means = np.random.uniform(0.0, np.max(I), (k, d)).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)

    for it in range(100):
        counts *= 0
        sums *= 0
        for x in range(h):
            for y in range(w):
                dist_best = 0.0
                k_best = 0
                for m in range(k):
                    dist_m = np.sum(np.square(means[m] - I[x, y, :]))
                    if dist_m < dist_best or m == 0:
                        k_best = m
                        dist_best = dist_m
                k_map[x, y] = k_best
                sums[k_best, :] += I[x, y, :]
                counts[k_best] += 1

        for m in range(k):
            if counts[m] > 0:
                means[m] = np.copy(sums[m, :] / counts[m])

        if np.sum(k_map - k_map_) == 0:
            if it == 0:
                means = np.random.uniform(0.0, np.max(I), (k, d)).astype(np.float64)
            break
        k_map_ = np.copy(k_map)

    J = np.zeros_like(I)
    for x in range(h):
        for y in range(w):
            J[x, y, :] = means[k_map[x, y]]

    return J


# ========================= IMAGE I/O ===========================================


def imread3D(path):
    """Reads an image from disk. Returns the array representation.

        Parameters
        ----------
        path : str
            Path to file (including file extension)

        Returns
        -------
        out : ndarray of float64
            Image as 3D array

        Notes
        -----
        'I' will always have 3 dimensions: (rows, columns dimensions).
        Last dimension will be of length 1 or 3, depending on the image.

    """
    I = np.array(imageio.imread(path).astype(np.float64))  # first use imread() from imageio
    if I.ndim == 2:
        h, w = I.shape
        I = I.reshape((h, w, 1)).astype(np.float64)  # if image has two dimensions, we add one dimension
    else:
        if np.all(I[:, :, 0] == I[:, :, 1]) and np.all(I[:, :, 0] == I[:, :, 2]):
            return I[:, :, 0:1:].astype(np.float64)
        h, w, d = I.shape
        if d == 4:
            I = I[:, :, :3]  # if image has 3 dimensions and 4 channels, drop last channel

    return I


def imsave3D(path, I):
    """Saves the array representation of an image to disk.

        Parameters
        ----------
        path : str
            Path to file (including file extension)
        I : ndarray of float64
            Array representation of an image

        Returns
        -------
        out : none

        Notes
        -----
        The given array must have 3 dimensions,
        where the length of the last dimension is either 1 or 3.
    """
    assert I.ndim == 3, "image to save must have three dimensions!"
    h, w, d = I.shape
    assert d in {1, 3}, "depth of image to save must be 1 or 3!"
    I_uint8 = I.astype(np.uint8)
    if d == 1:
        I_uint8 = I_uint8.reshape(h, w)
    imageio.imsave(path, I_uint8)


def imshow3D(*I):
    """Shows the array representation of one or more images in a jupyter notebook.

        Parameters
        ----------
        I : ndarray of float64
            Array representation of an image
            Concatenates multiple images

        Returns
        -------
        out : none

        Notes
        -----
        The given array must have 3 dimensions,
        where the length of the last dimension is either 1 or 3.
    """

    if len(I) == 1:
        I = I[0]
    else:
        channels = [i.shape[2] for i in I]
        heights = [i.shape[0] for i in I]
        max_height = max(heights)
        max_channels = max(channels)

        if min(channels) != max_channels:  # if one image has three channels ..
            I = list(I)
            for i in range(len(I)):
                dim = channels[i]
                if dim == 1:  # .. and another has one channel ..
                    I[i] = np.dstack((I[i], I[i], I[i]))  # .. expand that image to three channels!

        if min(heights) != max_height:  # if heights of some images differ ..
            I = list(I)
            for i in range(len(I)):
                h, w, d = I[i].shape
                if h < max_height:  # .. expand by 'white' rows!
                    I_expanded = np.ones((max_height, w, d), dtype=np.float64) * 255
                    I_expanded[:h, :, :] = I[i]
                    I[i] = I_expanded

        seperator = np.ones((max_height, 3, max_channels), dtype=np.float64) * 255
        seperator[:, 1, :] *= 0
        I_sep = []
        for i in range(len(I)):
            I_sep.append(I[i])
            if i < (len(I) - 1):
                I_sep.append(seperator)
        I = np.hstack(I_sep)  # stack all images horizontally

    assert I.ndim == 3
    h, w, d = I.shape
    assert d in {1, 3}
    if d == 1:
        I = I.reshape(h, w)
    IPython.display.display(PIL.Image.fromarray(I.astype(np.ubyte)))
