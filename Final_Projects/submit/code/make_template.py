import cv2
import numpy as np


if __name__ == '__main__':
    # 在线拾取坐标https://uutool.cn/img-coord/
    template_path = '../data/templates/'
    # # English
    # raw_template = cv2.imread(template_path + 'typed_raw/symbol.png', cv2.IMREAD_GRAYSCALE)[:, 77: 166]
    # template_h = 86
    # # Chinese
    raw_template = cv2.imdecode(np.fromfile(template_path + 'typed_raw/J-S.png', dtype=np.uint8), -1)[:, 1015: 1092]
    template_h = 73
    template_w = int(template_h / raw_template.shape[0] * raw_template.shape[1] * 1)
    template = cv2.resize(raw_template, (template_w, template_h))
    cv2.imwrite(template_path + f'typed/data/template.png', template)
