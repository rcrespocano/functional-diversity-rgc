# -*- coding: utf-8 -*-

import imageio
import cv2


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end='\r')

    # Print New Line on Complete
    if iteration == total:
        print()


def generate_gif(filename, array, fps=30):
    array /= array.max()
    array *= 255.0
    array = array.astype('uint8')
    imageio.mimwrite(filename, array, fps=30)


def enlarge_image(reduced_image, image_width, image_height):
    return cv2.resize(reduced_image, (image_width, image_height), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
