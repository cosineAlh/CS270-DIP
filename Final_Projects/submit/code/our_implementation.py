import cv2
import numpy as np

"""
by ALH
"""

def matchTmpl(img, tmpl):
    # 对cv2.matchTemplate的实现
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

    img_H = img.shape[0]
    img_W = img.shape[1]
    tmpl_H = tmpl.shape[0]
    tmpl_W = tmpl.shape[1]

    R = np.zeros((img_H - tmpl_H + 1, img_W - tmpl_W + 1)).astype(np.float32)

    tmpl_avg = np.mean(tmpl)

    for i in range(0, img_H - tmpl_H + 1):
        for j in range(0, img_W - tmpl_W + 1):
            img_avg = np.mean(img[i:i + tmpl_H, j:j + tmpl_W])
            tmp_T = np.array(tmpl) - tmpl_avg
            tmp_I = np.array(img[i:i + tmpl_H, j:j + tmpl_W]) - img_avg
            tmp_T2 = tmp_T ** 2
            tmp_I2 = tmp_I ** 2

            corr_ab = sum(sum(tmp_I * tmp_T))
            sq = np.sqrt(sum(sum(tmp_T2)) * sum(sum(tmp_I2)))
            if sq == 0:
                R[i][j] = 0
            else:
                corr = corr_ab / sq
                R[i][j] = corr

    return R


from utils import Canny

def findContours(binary):
    # 对cv2.findContours的实现
    height, width = binary.shape[:2]
    visited = np.zeros((height, width), dtype=np.uint8)

    contours = []
    for y in range(height):
        for x in range(width):
            # if current point is contour and is not been visited
            if binary[y, x] == 255 and visited[y, x] == 0:
                contour = []
                current_point = (x, y)

                contour.append(current_point)
                visited[y, x] = 255

                while True:
                    nx, ny = getNeighbor(current_point)

                    # check boundary
                    if 0 <= nx < width and 0 <= ny < height:
                        if binary[ny, nx] == 255 and visited[ny, nx] == 0:
                            current_point = (nx, ny)
                            contour.append(current_point)
                            visited[ny, nx] = 255
                            continue
                    break
                contours.append(np.array(contour))
    return contours

def getNeighbor(point):
    x, y = point
    neighbors = [(x, y-1), (x-1, y), (x+1, y), (x, y+1)]
    return neighbors[0]
