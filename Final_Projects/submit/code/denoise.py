import cv2
import numpy as np
from utils import show, Canny


def open_op(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    dst = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return dst / 255


def get_color(image):
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 查询HSV色表得到黑白的minmax
    low_hsv = np.array([0, 0, 0])
    high_hsv = np.array([255, 255, 80])
    mask = (cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv) == 255)

    out = np.zeros(image.shape[:2])
    out[mask] = image_grey[mask]
    return out / 255


def poetry_denoise(original_poetry, out_path):
    mask = get_color(original_poetry)
    poetry = np.ones_like(original_poetry) * np.median(original_poetry)
    poetry[mask > 0] = original_poetry[mask > 0]
    poetry = poetry.astype(np.uint8)

    # Dilation
    kernel = np.ones((2, 2), np.uint8)
    poetry = 255 - cv2.dilate(255 - poetry, kernel, iterations=1)

    cv2.imwrite(out_path + 'chinese_binary.png', poetry)
    print('Chinese denoised, remove background lines')


def handwritten_denoise(original, out_path):
    _, binary = cv2.threshold(original, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    H = original.shape[0]
    W = original.shape[1]

    dst = open_op(binary)
    kernel = np.ones((3, 3), np.uint8)
    dst = cv2.dilate(dst, kernel, iterations=1)
    binary = 255 - binary

    for i in range(0, H):
        for j in range(0, W):
            if dst[i][j]:
                if binary[i - 5][j] or binary[i + 5][j] or binary[i + 4][j]:
                    binary[i][j] = 255

    cv2.imwrite(out_path + 'english_binary.png', binary)
    print('English denoised, remove unlines and did some repair')


if __name__ == '__main__':
    original_english = cv2.imread('../data/original/handwritten_English.jpg', cv2.IMREAD_GRAYSCALE)
    original_chinese = cv2.imread('../data/temp/sheared.png', cv2.IMREAD_GRAYSCALE)
    temp_path = '../data/temp/'
    poetry_denoise(original_chinese, temp_path)
    handwritten_denoise(original_english, temp_path)
