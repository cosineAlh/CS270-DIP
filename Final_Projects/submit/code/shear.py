import cv2
import numpy as np
from utils import Canny, linear_normalize, show
import tqdm


def Hough(image, theta_size):
    rho_size = round(np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2) * 2)
    hough_graph = np.zeros((rho_size, theta_size))
    print('Computing hough ... ')
    for x in tqdm.trange(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x, y] != 0:
                for t in range(theta_size):
                    theta = t / theta_size * np.pi - np.pi / 2
                    rho = round(x * np.cos(theta) + y * np.sin(theta) + (rho_size - 1) / 2)
                    hough_graph[rho, t] += 1
    return hough_graph


def get_angle(original, out_path):
    theta_size = 2000

    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    edges = Canny(original)
    hough = Hough(edges, theta_size)
    hough = linear_normalize(hough).astype(np.uint8)
    cv2.imwrite(out_path + 'hough.png', hough)

    masked_hough = hough.copy()
    masked_hough[:, 1000 - 50: 1000 + 50] = 0
    masked_hough[:, 0] = 0
    masked_hough[:, 1999] = 0
    masked_hough[:, 1500] = 0
    masked_hough[:, 500] = 0

    max_rho, max_t = np.unravel_index(np.argmax(masked_hough), masked_hough.shape)
    max_theta = max_t / theta_size * 180 - 90
    masked_hough[:, max_t] = 255

    cv2.imwrite(out_path + 'masked_hough.png', masked_hough)
    return max_theta


def shear(original, out_path, axis='x'):
    assert axis in ['x', 'y']
    angle = get_angle(original, out_path)
    print(f'found angle = {angle}, shear with this angle')
    if axis == 'x':
        M = np.array([[1, np.tan((90 - angle) / 180 * np.pi), 0],
                      [0, 1, 0]]).astype(np.float32)
    else:
        M = np.array([[1, 0, 0],
                      [np.tan((90 - angle) / 180 * np.pi), 1, 0]]).astype(np.float32)
    H, W = original.shape[:2]
    sheared = cv2.warpAffine(original, M, (H * 2, int(W * 0.65)), borderValue=np.median(original, axis=(0, 1)))

    cv2.imwrite(f'../data/temp/sheared.png', sheared)


if __name__ == '__main__':
    # a_shear()
    # b_shear()
    # c_shear()
    original = cv2.imread('../data/original/handwritten_English.jpg', cv2.IMREAD_GRAYSCALE)
    out_path = '../data/temp/'
    shear(original, out_path)
