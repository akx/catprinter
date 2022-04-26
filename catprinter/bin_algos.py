from math import ceil

import cv2
import numpy as np

registry = {}


def get_algo_ids():
    return list(registry.keys())


def bin_algo(id: str, name: str):
    def decorator(func):
        registry[id] = (name, func)
        return func

    return decorator


@bin_algo('floyd-steinberg', 'Floyd-Steinberg dithering')
def floyd_steinberg_dither(img):
    '''Applies the Floyd-Steinberg dithering to img.
    img is expected to be a 8-bit grayscale image.

    Algorithm borrowed from wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering.
    '''
    img = img.copy()
    h, w = img.shape

    def adjust_pixel(y, x, delta):
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        img[y][x] = min(255, max(0, img[y][x] + delta))

    for y in range(h):
        for x in range(w):
            new_val = 255 if img[y][x] > 127 else 0
            err = img[y][x] - new_val
            img[y][x] = new_val
            adjust_pixel(y, x + 1, err * 7 / 16)
            adjust_pixel(y + 1, x - 1, err * 3 / 16)
            adjust_pixel(y + 1, x, err * 5 / 16)
            adjust_pixel(y + 1, x + 1, err * 1 / 16)
    return (img > 127)


@bin_algo('halftone', 'Halftone dithering')
def halftone_dither(img):
    '''Applies Haltone dithering using different sized circles

    Algorithm is borrowed from https://github.com/GravO8/halftone
    '''

    def square_avg_value(square):
        '''
        Calculates the average grayscale value of the pixels in a square of the
        original image
        Argument:
            square: List of N lists, each with N integers whose value is between 0
            and 255
        '''
        sum = 0
        n = 0
        for row in square:
            for pixel in row:
                sum += pixel
                n += 1
        return sum / n

    side = 4
    jump = 4  # Todo: make this configurable
    alpha = 3
    height, width = img.shape

    if not jump:
        jump = ceil(min(height, height) * 0.007)
    assert jump > 0, "jump must be greater than 0"

    height_output, width_output = side * ceil(height / jump), side * ceil(width / jump)
    canvas = np.zeros((height_output, width_output), np.uint8)
    output_square = np.zeros((side, side), np.uint8)
    x_output, y_output = 0, 0
    for y in range(0, height, jump):
        for x in range(0, width, jump):
            output_square[:] = 255
            intensity = 1 - square_avg_value(img[y:y + jump, x:x + jump]) / 255
            radius = int(alpha * intensity * side / 2)
            if radius > 0:
                # draw a circle
                cv2.circle(
                    output_square,
                    center=(side // 2, side // 2),
                    radius=radius,
                    color=(0, 0, 0),
                    thickness=-1,
                    lineType=cv2.FILLED
                )
            # place the square on the canvas
            canvas[y_output:y_output + side,
            x_output:x_output + side] = output_square
            x_output += side
        y_output += side
        x_output = 0
    return (canvas > 127)


@bin_algo('mean-threshold', 'Mean Threshold')
def mean_threshold_dither(resized):
    return resized > resized.mean()
