#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import logging
import os
import cv2
import numpy as np


def preprocess(img):
    height, width = img.shape
    img_gaussian = cv2.GaussianBlur(img, (height // 10 * 8 + 21, width // 10 * 8 + 21), 0)
    img = (img / img_gaussian * 255)
    img = np.select([img <= 255], [img], default=255).astype('uint8')
    return img


def getRoughLine(img_gray):
    # 最小行宽  10
    # 判断为一行的空白的最大距离  1
    min_line_width = 20
    max_blank_dis = 1

    # TODO:限定范围 左右多少范围内不作为分行依据
    index_thre = 0
    img = img_gray[:, index_thre:1200 - index_thre]

    # 二值化（像素大于100的点均为白色）
    ret2, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    line_index = np.unique(np.where(img == 0)[0])
    begin_index = []
    end_index = []
    begin = line_index[0]
    for i in range(len(line_index) - 1):
        if line_index[i + 1] - line_index[i] > max_blank_dis:
            end = line_index[i]
            begin_index.append(begin)
            end_index.append(end)
            begin = line_index[i + 1]

    end = line_index[-1]
    begin_index.append(begin)
    end_index.append(end)

    # print(begin_index, end_index)

    new_begin_index = []
    new_end_index = []
    for i in range(len(begin_index)):
        if end_index[i] - begin_index[i] > min_line_width:
            new_begin_index.append(begin_index[i])
            new_end_index.append(end_index[i])

    return new_begin_index, new_end_index


def get_accurate_line(img_gray, up, line, down):
    grayNot = cv2.bitwise_not(img_gray)

    threImg = cv2.threshold(grayNot, 100, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)[1]
    kernel = np.ones((20, 20), np.uint8)  # 定义核。无符号8位。
    # result = cv2.erode(threImg,kernel)       #腐蚀
    result = cv2.dilate(threImg, kernel)
    # 输出腐蚀图
    # cv2.imwrite('special.jpg', result)
    # ret2, img = cv2.threshold(imGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    output = cv2.connectedComponentsWithStats(result.astype(np.uint8), 4)
    num_labels = output[0]
    label_image = output[1]  # Image with a unique label for each connected region
    stats = output[2]
    centroids = output[3]  # Centroid indices for each connected region

    # 直线  穿过非背景连通域  确定连通域上下界 判断是否过界
    # 找到背景对应的label
    label_bg = np.argmax(stats.T[-1])
    # 这一行中除背景外的其它点坐标

    index_l = []
    for index, label in enumerate(label_image[line]):
        # 限定范围 左右多少范围内不作为分行依据
        index_thre = 50
        if index < index_thre:
            continue
        elif index > 1200-index_thre:
            break
        # 判断是否是背景对应的连通域
        if label != label_bg:
            index_l.append(index)

    # 需要处理的区域
    areas = []
    # 当前x
    current_x = 0
    # flag 标注是否是一行
    flag_one_line = 0
    for i in range(len(index_l)):
        index = index_l[i]
        # 如果index 在当前x前则认为已经被加入到处理区域中
        if index <= current_x:
            continue
        label = label_image[line][index]
        x, y, w, h, s = stats[label]
        # 比较连通域中线和分割线位置
        if y + h / 2 > line:
            # 连通域中线大于分割线 说明 属于下半部
            up_or_down = 0
            # 新分割线是连通域上界
            new_line = y
        else:
            up_or_down = 1
            new_line = y + h

        # 判断是否越界
        if up_or_down:
            # 属于上半部  新分割线大于下界 说明越界 越界则直接停止 认为是一行
            if new_line > down:
                flag_one_line = 1
                break
        else:
            if new_line < up:
                flag_one_line = 1
                break

        # 更改当前x
        current_x = x + w
        areas.append((up_or_down, new_line, x, x + w))

    if flag_one_line:
        return 1, 0, 0
    else:

        img_up = np.ones(img_gray.shape, np.uint8) * 255
        img_down = np.ones(img_gray.shape, np.uint8) * 255
        img_up[0:line, :] = img_gray[0:line, :]
        img_down[line + 1:, :] = img_gray[line + 1:, :]

        for area in areas:
            if area[0]:
                img_up[line:area[1] + 1, area[2]:area[3]] = img_gray[line:area[1] + 1, area[2]:area[3]]
                img_down[line + 1:area[1] + 1, area[2]:area[3]] = np.ones((area[1] - line, area[3] - area[2])) * 255

            else:
                img_up[area[1]:line, area[2]:area[3]] = np.ones((line - area[1], area[3] - area[2])) * 255
                img_down[area[1]:line + 1, area[2]:area[3]] = img_gray[area[1]:line + 1, area[2]:area[3]]

        return 0, img_up, img_down


def get_boundary_vertical(gray):
    # ret2, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m, n = np.where(gray != 255)
    if m.size != 0:
        top, down = np.min(m), np.max(m)
    else:
        top, down = 0, gray.shape[0] - 1
    # if n.size != 0:
    #     left, right = np.min(n), np.max(n)
    # else:
    #     left, right = 0, gray.shape[1] - 1

    return gray[top:down + 1:]


def getBoundary(gray):

    # ret2, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    m, n = np.where(gray != 255)
    if m.size != 0:
        top, down = np.min(m), np.max(m)
    else:
        top, down = 0, gray.shape[0] - 1
    if n.size != 0:
        left, right = np.min(n), np.max(n)
    else:
        left, right = 0, gray.shape[1] - 1

    return gray[:, left:right + 1]


def get_lines(img):
    height, width = img.shape
    # 8.分行
    # 粗分行
    begin, end = getRoughLine(img)
    # return begin,end
    line_num = len(begin)
    # print(line_num)
    # 8.1 精细分行
    img_l = []
    lines = []
    # 设阈值、 峰值判断 等 猜测存在多行

    for index in range(line_num):
        img_line = img[begin[index]:end[index] + 1, :]
        # x轴投影小于40个点认为不是一行
        ret2, img8 = cv2.threshold(img_line, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        x_index = np.unique(np.where(img8 == 0)[1])
        if len(x_index) < 40:
            continue
        img_l.append(get_boundary_vertical(img_line))
    return img_l, begin, end, lines


def get_word(img,language):
    # 最小行宽  10
    # 判断为一行的空白的最大距离  1
    if language == 'en':
        # 最小宽度 用于处理完后进行筛选 小于的扔掉 主要筛选标点
        min_line_width = 10
        # 最小间隔 下一个分割小大于它，当前的才能认为一个词的结束
        min_blank_dis = 6
        # 每个投影剪掉一个基本单位 也就是剪掉下划线的尺寸
        cut_off_num = 4

        # 割6 // 7
        img = img[:img.shape[0] * 6 // 7, :]
        ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.erode(img,(100,100))
    elif language == 'ch':
        # 最小宽度 用于处理完后进行筛选 小于的扔掉  主要筛选标点
        min_line_width = 20
        # 最小间隔 下一个分割小大于它，当前的才能认为一个词的结束
        min_blank_dis = 7
        # 每个投影剪掉一个基本单位 也就是剪掉下划线的尺寸
        cut_off_num = 0

    else:
        raise '没有这种语言'


    # # TODO:限定范围 左右多少范围内不作为分行依据
    # index_thre = 0
    # img = imGray[:,index_thre:1200-index_thre]

    # 二值化（像素大于100的点均为白色）
    ret2, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    line_index, num = np.unique(np.where(img == 0)[1],return_counts=True)

    line_index = line_index[np.where(num>cut_off_num)]
    begin_index = []
    end_index = []
    begin = line_index[0]
    for i in range(len(line_index) - 1):
        if line_index[i + 1] - line_index[i] > min_blank_dis:
            end = line_index[i]
            begin_index.append(begin)
            end_index.append(end)
            begin = line_index[i + 1]

    end = line_index[-1]
    begin_index.append(begin)
    end_index.append(end)

    new_begin_index = []
    new_end_index = []
    for i in range(len(begin_index)):
        if end_index[i] - begin_index[i] > min_line_width:
            new_begin_index.append(begin_index[i])
            new_end_index.append(end_index[i])

    return new_begin_index, new_end_index


def split_words(img,language):
    height, width = img.shape
    # 8.分行
    # 粗分行
    left, right = get_word(img, language)
    # return begin,end
    word_num = len(left)
    # print(line_num)

    img_l = []
    for index in range(word_num):
        img_word = img[:, left[index]:right[index] + 1]
        # # 行投影
        # point_nums = getLineProjection(img_line)
        #
        # # 找峰谷
        # peaks_num, peaks, downs = find_peak(point_nums)
        # # 遍历波谷找三条线
        # for i in range(peaks_num-1):
        #     line = downs[i]
        #     up = peaks[i]
        #     down = peaks[i+1]
        #     flag_one_line, img_up, img_down = getAccurateLine(img_line, up, line, down)
        #
        #     if flag_one_line:
        #         pass
        #     else:
        #         img_l.append(getBoundary(img_up))
        #         img_line = img_down
        #         lines.append(line+begin[index])
        #
        img_l.append(getBoundary(img_word))
    return img_l, left, right


def segment(img, img2, output_path_line, output_path_word, language='en'):

    if not os.path.exists(output_path_line):
        os.makedirs(output_path_line)
    if not os.path.exists(output_path_word):
        os.makedirs(output_path_word)

    # img = cv2.imread(file_path, 0)
    cv2.imwrite(output_path_line + '/gray.png', img)

    # preprocess
    original = img.copy()
    img = preprocess(img)
    # img = cv2.imread(file_path,0)
    cv2.imwrite(output_path_line + '/pre.png', img)

    # split_lines
    line_img_list, begin, end, lines = get_lines(img)
    line_num = len(line_img_list)
    print(f'line_num:{line_num}')

    # 画线
    h, w = img.shape
    img_draw_lines = img.copy()
    for index in range(len(begin)):
        cv2.line(img_draw_lines, (0, begin[index]), (w, begin[index]), 0)
        cv2.line(img_draw_lines, (0, end[index]), (w, end[index]), 0)
    cv2.imwrite(output_path_line + '/get_lines.png', img_draw_lines)

    # img2 = cv2.imread(file_path2, 0)
    line_img_list2 = []
    word_img_list2 = []

    # split_word
    for i in range(line_num):
        img = line_img_list[i]
        # split_words
        word_img_list, left, right = split_words(img, language)
        word_num = len(word_img_list)

        # 画线，保存行
        h, w = img.shape
        img_draw_word = img.copy()
        for index in range(len(left)):
            cv2.line(img_draw_word, (left[index], 0), (left[index], h), 0)
            cv2.line(img_draw_word, (right[index], 0), (right[index], h), 0)
        path = output_path_line + '/' + 'line' + str(i) + '.png'
        cv2.imwrite(path, img_draw_word)
        print(f'line_index:{i},word_num:{word_num}')

        if language == 'en':
            pad = 20  # 上下延长一部分留白
        else:
            pad = 10
        line_img = img2[begin[i]-pad:end[i]+1+pad, :]
        line_img_list2.append(line_img)

        # 保存词
        for j in range(len(left)):
            if language == 'en':
                pad = 7     # 左右延长一部分留白
                word_img = line_img[:, left[j]-pad:right[j]+1+pad]
            else:
                center_x = (right[j] + left[j]) // 2
                half_x = line_img.shape[0] // 2
                word_img = line_img[:, center_x - half_x: center_x + half_x]

            word_img_list2.append(word_img)
            path = output_path_word + '/' + 'word' + str(i) + '_' + str(j) + '.png'
            cv2.imwrite(path, word_img)
            # cv2.imshow('',word_img)
            # cv2.waitKey()


if __name__ == '__main__':
    # segment(分割依据，分割图片，输出图片，语言） 语言可以不分都用en
    segment('./img/p1.jpg', './img/p2.jpg', 'output/p2', 'en')
    # segment('./img/ch.jpg','./img/ch.jpg','output/ch','ch')
