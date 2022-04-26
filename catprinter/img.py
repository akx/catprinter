import cv2

from catprinter import logger
from catprinter.bin_algos import registry


def read_img(
    filename,
    print_width,
    img_binarization_algo,
):
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im = apply_binarization(im, img_binarization_algo, print_width)
    # Invert the image before returning it.
    return ~im


def apply_binarization(im, algorithm, print_width):
    height, width = im.shape
    if algorithm == 'none':
        if width != print_width:
            raise RuntimeError(
                f'Wrong width of {width} px. '
                f'An image with a width of {print_width} px '
                f'is required for "none" binarization'
            )
        return (im > 127)

    try:
        name, binarizer = registry[algorithm]
    except KeyError:
        raise RuntimeError(
            f'unknown image binarization algorithm: '
            f'{algorithm}'
        )

    factor = print_width / width
    new_size = (int(width * factor), int(height * factor))
    logger.info(f'⏳ Resizing image to {new_size}...')
    im = cv2.resize(im, new_size, interpolation=cv2.INTER_AREA)
    logger.info(f'⏳ Applying {name} to image...')
    im = binarizer(im)
    logger.info('✅ Done.')
    return im


def show_preview(bin_img):
    # Convert from our boolean representation to float and invert.
    preview_img = ~bin_img.astype(float)
    cv2.imshow('Preview', preview_img)
    logger.info('ℹ️  Displaying preview.')
    # Calling waitKey(1) tells OpenCV to process its GUI events and actually display our image.
    cv2.waitKey(1)
    if input('🤔 Go ahead with print? [Y/n]? ').lower() == 'n':
        raise RuntimeError('Aborted print.')
