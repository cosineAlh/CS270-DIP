import numpy as np
from utils import show, nms_multiclass, plot_bboxes
import cv2
import matplotlib.pyplot as plt
import os
import tqdm
from our_implementation import matchTmpl

# reference: https://www.cnblogs.com/leixiaohua1020/p/3901903.html
# TODO: 考虑 Fourier Descriptor (需要分割单个单词)
# TODO: 考虑 Dynamic Time Warping
# TODO: 考虑 Chain Code


# def template_match_old(img_src, img_templ):
#
#     plt.rc('font', family='Youyuan', size='9')
#     plt.rc('axes', unicode_minus='False')
#     plt.rcParams['figure.figsize']=(120, 60)
#     print('img_src.shape:', img_src.shape)
#     print('img_templ.shape:', img_templ.shape)
#
#     lt = [0.9999, 0.23, 0.9999, 0.98, 0.999, 0.8]  # 0.75
#     # for method in [1, 3, 5]:
#     for method in [cv2.TM_CCOEFF_NORMED]:
#         # 模板匹配
#         result_t = cv2.matchTemplate(img_src, img_templ, method)
#         # 筛选大于一定匹配值的点
#         val, result = cv2.threshold(result_t, lt[method], 1.0, cv2.THRESH_BINARY)
#         match_locs = cv2.findNonZero(result)
#         if match_locs is None:
#             print(f'match_locs.shape of method {method}:', (0, 1, 2))
#             continue
#         print(f'match_locs.shape of method {method}:', match_locs.shape)
#
#         img_disp = img_src.copy()
#         for match_loc_t in match_locs:
#             # match_locs是一个3维数组，第2维固定长度为1，取其下标0对应数组
#             match_loc = match_loc_t[0]
#             # 注意计算右下角坐标时x坐标要加模板图像shape[1]表示的宽度，y坐标加高度
#             right_bottom = (match_loc[0] + img_templ.shape[1], match_loc[1] + img_templ.shape[0])
#             # print('match_loc:', match_loc)
#             # print('result_t:', result_t[match_loc[1], match_loc[0]])
#             # 标注位置
#             cv2.rectangle(img_disp, match_loc, right_bottom, (0, 255, 0), 5, 8, 0)
#             cv2.circle(result, match_loc, 10, (255, 0, 0), 3)
#
#         # 显示图像
#         fig, ax = plt.subplots(2, 2)
#         fig.suptitle(f'多目标匹配 method{method}')
#         ax[0, 0].set_title('img_src')
#         ax[0, 0].imshow(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB))
#         ax[0, 1].set_title('img_templ')
#         ax[0, 1].imshow(cv2.cvtColor(img_templ, cv2.COLOR_BGR2RGB))
#         ax[1, 0].set_title('result')
#         ax[1, 0].imshow(result, 'gray')
#         ax[1, 1].set_title('img_disp')
#         ax[1, 1].imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
#         # ax[0,0].axis('off');ax[0,1].axis('off');ax[1,0].axis('off');ax[1,1].axis('off')
#
#         plt.savefig(f'../temp/method {method}.png')
#         # plt.show()


def shear_image(image, shear_x, shear_y=0):
    height, width = image.shape[:2]
    center_x, center_y = width / 2, height / 2

    # Step 3: Translate to center
    translate_to_origin = np.array([[1, 0, -center_x],
                                    [0, 1, -center_y],
                                    [0, 0, 1]])

    # Step 4: Apply shear
    shear_matrix = np.array([[1, shear_x, 0],
                             [shear_y, 1, 0],
                             [0, 0, 1]])

    # Step 5: Translate back to original position
    translate_back = np.array([[1, 0, center_x],
                               [0, 1, center_y],
                               [0, 0, 1]])

    transformation_matrix = np.matmul(translate_back, np.matmul(shear_matrix, translate_to_origin))

    # Perform the affine transformation
    sheared_image = cv2.warpAffine(image, transformation_matrix[:2], (width, height),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return sheared_image


def elongate_image(image, ratio):
    # 用于处理字符g
    height, width = image.shape[:2]
    t1, t2 = 0.75, 0.83
    a, b, c = image[:int(height * t1), :], image[int(height * t1): int(height * t2), :], image[int(height * t2):, :]
    b = cv2.resize(b, (width, int((t2-t1) * height * ratio)))
    out_image = np.concatenate((a, b, c), axis=0)

    return out_image


# def Kullback_Leibler(x, y):
#     x = x / np.sum(x)
#     y = y / np.sum(y)
#     kl = 0.0
#     for i in range(10):
#         kl += x[i] * np.log(x[i] / y[i])
#     return kl


# def calculate_sift_distance(image1, image2):
#     # Initialize SIFT detector
#     # sift = cv2.SIFT_create()
#     # fast = cv2.FastFeatureDetector_create()
#     # brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#     orb = cv2.ORB_create()
#     # Detect keypoints and compute descriptors for image 1
#     # keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
#     # keypoints1, descriptors1 = computeKeypointsAndDescriptors(image1)
#     # keypoints1 = fast.detect(image1, None)
#     # keypoints1, descriptors1 = brief.compute(image1, keypoints1)
#     keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
#     # Detect keypoints and compute descriptors for image 2
#     # keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
#     # keypoints2, descriptors2 = computeKeypointsAndDescriptors(image2)
#     # keypoints2 = fast.detect(image2, None)
#     # keypoints2, descriptors2 = brief.compute(image2, keypoints2)
#     keypoints2, descriptors2 = orb.detectAndCompute(image1, None)
#     if len(keypoints1) <= 1 or len(keypoints2) <= 1:
#         return 1.0
#     # Create a BFMatcher object
#     bf = cv2.BFMatcher()
#     # Match descriptors using the BFMatcher
#     matches = bf.knnMatch(descriptors1, descriptors2, k=2)
#     # Apply ratio test to filter good matches
#     good_matches = []
#     if len(matches) != 0:
#         for m, n in matches:
#             if m.distance < 0.75 * n.distance:
#                 good_matches.append(m)
#     # Calculate the distance based on the number of good matches
#     distance = 1.0 - (len(good_matches) / max(len(descriptors1), len(descriptors2)))
#     return distance


# def calculate_gradient(image):
#     gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
#     gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
#     magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y)
#     return magnitude, angle
#
# def calculate_histogram(angle, magnitude, bins):
#     histogram = np.zeros((angle.shape[0], angle.shape[1], bins))
#
#     for i in range(angle.shape[0]):
#         for j in range(angle.shape[1]):
#             hist, _ = np.histogram(angle[i, j], bins=bins, range=(0, 2 * np.pi), weights=magnitude[i, j])
#             histogram[i, j, :] = hist
#     histogram = cv2.normalize(histogram, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     return histogram.astype(np.uint8)
#
# def calculate_hog(image, template, bins):
#     # Calculate gradients for both template and image
#     template_magnitude, template_angle = calculate_gradient(template)
#     image_magnitude, image_angle = calculate_gradient(image)
#
#     # Calculate histograms
#     template_histogram = calculate_histogram(template_angle, template_magnitude, bins)
#     image_histogram = calculate_histogram(image_angle, image_magnitude, bins)
#
#     density_maps = []
#     for i in range(bins):
#         response = cv2.matchTemplate(image_histogram[:, :, i], template_histogram[:, :, i], cv2.TM_CCOEFF_NORMED)
#         density_map = np.zeros(image.shape)
#         density_map[0: response.shape[0], 0: response.shape[1]] = response
#         density_maps.append(density_map)
#
#     # Average the density maps
#     average_density_map = np.mean(density_maps, axis=0)
#
#     return average_density_map


class Recognizer:
    def __init__(self, image, template_path, temp_path, words_path, out_path, language):
        # self.original_image = None
        self.preprocessed_image = image
        # self.current = {'character': None,
        #                 'template': None,
        #                 'match_locations': None,
        #                 'density_map': None,
        #                 'best_location': None}
        self.typed_templates = {}
        self.image_templates = {}
        self.current = dict()
        self.all_bbox = []
        self.template_path = template_path
        self.temp_path = temp_path
        self.out_path = out_path
        self.words_path = words_path
        self.string = ''
        assert language in ['en', 'ch']
        self.language = language

        # height, width, shear, elongate
        if language == 'en':
            self.character_params = {
                 'e': {'h': 1.0, 'w': 1.0},
                 'o': {'h': 0.9, 'w': 0.6, 's': 0.1},##
                 'h': {'h': 1.0, 'w': 0.9},
                 't': {'h': 0.8, 'w': 1.0},#
                 'r': {'h': 1.0, 'w': 1.0},
                 'n': {'h': 0.9, 'w': 0.9},#
                 'l': {'h': 1.0, 'w': 1.0},
                 'y': {'h': 1.2, 'w': 0.9},#
                 'i': {'h': 0.8, 'w': 1.0},
                 'a': {'h': 0.8, 'w': 1.0},#
                 'w': {'h': 1.0, 'w': 0.8},
                 'u': {'h': 1.0, 'w': 0.9},
                 'd': {'h': 0.9, 'w': 1.0},
                 's': {'h': 1.0, 'w': 1.0},
                 'v': {'h': 1.0, 'w': 1.0},
                 'p': {'h': 1.0, 'w': 0.8},
                 't_cap': {'h': 0.8, 'w': 1.0},#
                 'b': {'h': 0.7, 'w': 1.0},#
                 'm': {'h': 1.0, 'w': 0.8},
                 'g': {'h': 0.8, 'w': 1.1, 'e': 3.3},##
                 'k': {'h': 0.8, 'w': 1.2},#
                 'c': {'h': 0.8, 'w': 1.2, 's': 0.2},##
                 'r_cap': {'h': 0.8, 'w': 1.0},#
                 'j': {'h': 1.0, 'w': 1.0},
                 'f': {'h': 1.3, 'w': 0.7},#
                 'y_cap': {'h': 1.0, 'w': 1.0},
                 '&': {'h': 0.7, 'w': 0.9, 's': 0.4}
                                     }
        else:
            self.character_params = {
                '芙': {'h': 1.0, 'w': 1.0},
                '蓉': {'h': 1.0, 'w': 1.0},
                '楼': {'h': 1.0, 'w': 1.0},
                '送': {'h': 1.0, 'w': 1.0},
                '辛': {'h': 1.0, 'w': 1.0},
                '渐': {'h': 1.0, 'w': 1.0},
                '唐': {'h': 1.0, 'w': 1.0},
                '代': {'h': 1.0, 'w': 1.0},
                '王': {'h': 1.0, 'w': 1.0},
                '昌': {'h': 1.0, 'w': 1.0},
                '龄': {'h': 1.0, 'w': 1.0},
                '寒': {'h': 1.0, 'w': 1.0},
                '雨': {'h': 1.0, 'w': 1.0},
                '连': {'h': 1.0, 'w': 1.0},
                '江': {'h': 1.0, 'w': 1.0},
                '夜': {'h': 1.0, 'w': 1.0},
                '入': {'h': 1.0, 'w': 1.0},
                '吴': {'h': 1.0, 'w': 1.0},
                '平': {'h': 1.0, 'w': 1.0},
                '明': {'h': 1.0, 'w': 1.0},
                '客': {'h': 1.0, 'w': 1.0},
                '楚': {'h': 1.0, 'w': 1.0},
                '山': {'h': 1.0, 'w': 1.0},
                '孤': {'h': 1.0, 'w': 1.0},
                '洛': {'h': 1.0, 'w': 1.0},
                '阳': {'h': 1.0, 'w': 1.0},
                '亲': {'h': 1.0, 'w': 1.0},
                '友': {'h': 1.0, 'w': 1.0},
                '如': {'h': 1.0, 'w': 1.0},
                '相': {'h': 1.0, 'w': 1.0},
                '问': {'h': 1.0, 'w': 1.0},
                '一': {'h': 1.0, 'w': 1.0},
                '片': {'h': 1.0, 'w': 1.0},
                '冰': {'h': 1.0, 'w': 1.0},
                '心': {'h': 1.0, 'w': 1.0},
                '在': {'h': 1.0, 'w': 1.0},
                '玉': {'h': 1.0, 'w': 1.0},
                '壶': {'h': 1.0, 'w': 1.0},
            }

    def preprocess_template(self):
        reference_h = 86
        character = self.current['character']
        template_h = int(reference_h * self.character_params[character]['h'])
        template_w = int(template_h / self.current['template'].shape[0] * self.current['template'].shape[1] *
                         self.character_params[character]['w'])
        self.current['template'] = cv2.resize(self.current['template'], (template_w, template_h))
        if 's' in self.character_params[character]:
            self.current['template'] = shear_image(self.current['template'], self.character_params[character]['s'])
        if 'e' in self.character_params[character]:
            self.current['template'] = elongate_image(self.current['template'], self.character_params[character]['e'])


    def visual_match_result(self):
        print(f'generating match result for {self.current["character"]}')
        plt.rc('font', family='Youyuan', size='9')
        plt.rc('axes', unicode_minus='False')
        plt.rcParams['figure.figsize'] = (120, 60)  # 显示匹配结果（高清）
        img_disp = self.preprocessed_image.copy()
        img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)
        density_disp = cv2.normalize(self.current['density_map'], None, 0, 255, cv2.NORM_MINMAX)
        density_disp = cv2.cvtColor(density_disp, cv2.COLOR_GRAY2BGR)
        for match_loc in self.current['match_locations']:
            # match_loc = self.current['best_location']
            right_bottom = (match_loc[0] + self.current['template'].shape[1],
                            match_loc[1] + self.current['template'].shape[0])
            # 标注
            cv2.rectangle(img_disp, match_loc, right_bottom, (0, 255, 0), 5, 8, 0)
            cv2.circle(density_disp, match_loc, 10, (255, 0, 0), 3)

        # 显示图像
        fig, ax = plt.subplots(2, 2)
        fig.suptitle(f'多目标匹配 method TM_CCOEFF_NORMED')
        ax[0, 0].set_title('img_src')
        ax[0, 0].imshow(cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2RGB))
        ax[0, 1].set_title('img_templ')
        ax[0, 1].imshow(cv2.cvtColor(self.current['template'], cv2.COLOR_BGR2RGB))
        ax[1, 0].set_title('density_map')
        ax[1, 0].imshow(self.current['density_map'], 'gray')
        ax[1, 1].set_title('img_disp')
        ax[1, 1].imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
        # ax[0,0].axis('off');ax[0,1].axis('off');ax[1,0].axis('off');ax[1,1].axis('off')

        plt.savefig(self.out_path + f'match_result_{self.current["character"]}.png')
        plt.close()

    def template_match(self, input_image, candidate_bound, threshold_rate=None, update=False):
        """
        candidate_bound: (min, max) output bbox counts
        threshold_rate: preserve with threshold with threshold * max_density
        option: 'update' use matched to update template
        """
        threshold = 0.75
        match_score = {}
        method = cv2.TM_CCOEFF_NORMED

        # 模板匹配
        # from matchTemp import matchTmpl
        self.current['density_map'] = matchTmpl(self.preprocessed_image, self.current['template'])
        # self.current['density_map'] = cv2.matchTemplate(input_image, self.current['template'], method)

        max_iter = 100
        iter = 0
        while len(match_score) < candidate_bound[0] or len(match_score) > candidate_bound[1]:
            if iter > max_iter:
                return
            iter += 1
            if len(match_score) < candidate_bound[0]:
                threshold *= 0.99
            elif len(match_score) > candidate_bound[1]:
                threshold *= 1.01
            # 阈值筛选
            val, density_threshold = cv2.threshold(self.current['density_map'], threshold, 1.0, cv2.THRESH_BINARY)
            self.current['match_locations'] = cv2.findNonZero(density_threshold)
            if self.current['match_locations'] is None:     # no match
                continue
            self.current['match_locations'] = self.current['match_locations'][:, 0, :]

            match_score = {}
            for idx, (match_y1, match_x1) in enumerate(self.current['match_locations']):
                match_x2 = match_x1 + self.current['template'].shape[0]
                match_y2 = match_y1 + self.current['template'].shape[1]
                candidate = input_image[match_x1: match_x2, match_y1: match_y2]
                # match_score[idx] = calculate_sift_distance(self.current['template'], candidate)
                match_score[idx] = self.current['density_map'][match_x1, match_y1]

            if threshold_rate is not None:
                max_density = np.max(self.current['density_map'])
                self.current['match_locations'] = \
                    np.array([[match_y1, match_x1] for (match_y1, match_x1) in self.current['match_locations'] if
                              self.current['density_map'][match_x1, match_y1] >= threshold_rate * max_density])

        if update:
            # best_candidate_idx = min(match_score, key=match_score.get)
            best_candidate_idx = 0
            self.current['best_location'] = self.current['match_locations'][best_candidate_idx, :]
            best_y1, best_x1 = self.current['match_locations'][best_candidate_idx, :]
            best_x2 = best_x1 + self.current['template'].shape[0]
            best_y2 = best_y1 + self.current['template'].shape[1]
            # slight modify (extend)
            x_center, y_center = (best_x2+best_x1)/2, (best_y2+best_y1)/2
            x_half, y_half = (best_x2-best_x1)/2, max((best_y2-best_y1)/2, 10)
            best_template = input_image[int(x_center - x_half*1.0): int(x_center + x_half*1.0),
                                                    int(y_center - y_half*0.9): int(y_center + y_half*0.9)]
            if self.language == 'en':
                cv2.imwrite(self.template_path + f'from_image/{self.current["character"]}.png', best_template)
            else:
                cv2.imencode('.png', best_template)[1].tofile(self.template_path + f'from_image/{self.current["character"]}.png')
            self.image_templates[self.current["character"]] = best_template
        # print(f'character: {self.current["character"]}, threshold = {threshold}')

    # def refine_templates(self):
    #     new_image_templates = {}
    #     for character in self.image_templates:
    #         self.image_templates[character]

    def recognize(self, full=True):
        """
        full=True 识别全图，但是不输出结果，而是输出全图bbox
        full=False 识别words_path下所有word图，并输出字符串
        """
        letter_freq = {
            'e': 0.130,
            'o': 0.075,
            'h': 0.061,
            't': 0.091,
            'r': 0.060,
            'n': 0.067,
            'l': 0.040,
            'y': 0.020,
            'i': 0.070,
            'a': 0.082,
            'w': 0.024,
            'u': 0.028,
            'd': 0.043,
            's': 0.063,
            'v': 0.010,
            'p': 0.019,
            't_cap': 0.020,     # given
            'b': 0.015,
            'm': 0.024,
            'g': 0.020,
            'k': 0.008,
            'c': 0.028,
            'r_cap': 0.010,     # given
            'j': 0.015,
            'f': 0.022,
            'y_cap': 0.010,     # given
            '&': 0.020        # given
        }
        if full:
            self.all_bbox = []
            for character in self.image_templates:
                e_bound = [1000, 2000]
                character_bound = [letter_freq[character] / letter_freq['e'] * x for x in e_bound]
                self.current = dict()
                self.current['template'] = self.image_templates[character]
                self.current['character'] = character
                self.template_match(input_image=self.preprocessed_image,
                                    candidate_bound=character_bound,
                                    threshold_rate=0.7)
                self.update_bbox()
            self.do_nms(threshold=0.15)     # 0.25
            plot_bboxes(self.preprocessed_image, self.all_bbox)
        else:
            self.string = ''
            file_list = sorted(os.listdir(self.words_path),
                               key=lambda x: tuple(map(int, x.replace('word', '').replace('.png', '').split('_'))))
            prev_x = None
            print('Recognizing each word ...')
            for name in tqdm.tqdm(file_list):
                word_image = cv2.imread(self.words_path + name, cv2.IMREAD_GRAYSCALE)
                word_image = np.pad(word_image, ((10, 10), (0, 0)), 'edge')  # pad, 用于处理最后一行和第一行可能的纵向距离过短
                x, y = map(int, name.replace('word', '').replace('.png', '').split('_'))
                if prev_x is not None and prev_x != x:
                    self.string += ' '     # \n
                elif prev_x is not None and self.language == 'en':
                    self.string += ' '
                prev_x = x

                self.all_bbox = []
                for character in self.image_templates:
                    e_bound = [20, 40]
                    character_bound = [letter_freq[character] / letter_freq['e'] * x for x in e_bound]
                    self.current = dict()
                    self.current['template'] = self.image_templates[character]
                    self.current['character'] = character
                    self.template_match(input_image=word_image, candidate_bound=character_bound, threshold_rate=0.7)
                    self.update_bbox()
                self.do_nms(threshold=0.15)  # 0.25
                self.all_bbox.sort(key=lambda x: x[1])      # 根据y1排序
                word_str = ''
                for bbox in self.all_bbox:
                    word_str += bbox[5]
                word_str = word_str.replace('r_cap', 'R').replace('t_cap', 'T').replace('y_cap', 'Y')
                if word_str == '&':      # 处理'&'
                    self.string = self.string[:-2]
                else:
                    self.string += word_str
                print('new word:', word_str)

    def update_bbox(self):
        # # 描述了字母的尺寸关系，高尺寸的压制低尺寸的
        # # 实施宗旨是为了便于识别，如果一个字母high recall low precision，其尺寸会被调低，比如大写字母
        # # 典型的例子是h压制n；u压制i
        # # 没有应用复杂度压制关系，例如e压制o, c
        # tier = {
        #     'e': 1,
        #     'o': 1,
        #     'h': 2,
        #     't': 1,
        #     'r': 0,
        #     'n': 1,
        #     'l': 1,
        #     'y': 2,
        #     'i': 0,
        #     'a': 1,
        #     'w': 2,
        #     'u': 1,
        #     'd': 2,
        #     's': 1,
        #     'v': 1,
        #     'p': 2,
        #     't_cap': 1,
        #     'b': 2,
        #     'm': 2,
        #     'g': 2,
        #     'k': 2,
        #     'c': 1,
        #     'r_cap': 1,
        #     'j': 2,
        #     'f': 1,
        #     'y_cap': 1,
        #     '&': 2
        # }
        h, w = self.current['template'].shape
        if self.current['match_locations'] is not None:
            for y1, x1 in self.current['match_locations']:
                x2 = x1 + h
                y2 = y1 + w
                score = self.current['density_map'][x1, y1]
                cls = self.current['character']
                # self.all_bbox.append([x1, y1, x2, y2, score, cls, tier[cls]])
                self.all_bbox.append([x1, y1, x2, y2, score, cls, 1])

    @staticmethod
    def calculate_distribution_distance(image1, image2):
        # # Normalize the histograms
        # # normalized_image1 = cv2.equalizeHist(image1)
        # # normalized_image2 = cv2.equalizeHist(image2)
        # normalized_image1 = image1
        # normalized_image2 = image2
        # # x-axis histogram
        # hist_x1 = np.sum(normalized_image1, axis=0)
        # hist_x2 = np.sum(normalized_image2, axis=0)
        # # y-axis histogram
        # hist_y1 = np.sum(normalized_image1, axis=1)
        # hist_y2 = np.sum(normalized_image2, axis=1)
        # from scipy.stats import wasserstein_distance
        # emd_x = wasserstein_distance(hist_x1, hist_x2)
        # emd_y = wasserstein_distance(hist_y1, hist_y2)
        # return emd_x, emd_y

        from sklearn.preprocessing import StandardScaler
        from scipy.stats import wasserstein_distance
        from scipy.special import kl_div
        hist_x1 = np.sum(image1, axis=0)
        hist_x2 = np.sum(image2, axis=0)
        hist_y1 = np.sum(image1, axis=1)
        hist_y2 = np.sum(image2, axis=1)

        l = np.linspace(0, 1, len(hist_x2))
        new_l = np.linspace(0, 1, len(hist_x1))
        hist_x2 = np.interp(new_l, l, hist_x2)
        l = np.linspace(0, 1, len(hist_y2))
        new_l = np.linspace(0, 1, len(hist_y1))
        hist_y2 = np.interp(new_l, l, hist_y2)

        hist_x1 = hist_x1 / np.sum(hist_x1)
        hist_x2 = hist_x2 / np.sum(hist_x2)
        hist_y1 = hist_y1 / np.sum(hist_y1)
        hist_y2 = hist_y2 / np.sum(hist_y2)

        scaler = StandardScaler()
        standardized_hist_x1 = scaler.fit_transform(np.array(hist_x1).reshape(-1, 1))
        standardized_hist_x2 = scaler.transform(np.array(hist_x2).reshape(-1, 1))
        scaler = StandardScaler()
        standardized_hist_y1 = scaler.fit_transform(np.array(hist_y1).reshape(-1, 1))
        standardized_hist_y2 = scaler.transform(np.array(hist_y2).reshape(-1, 1))

        # # Calculate the Wasserstein distance between the standardized lists
        # emd_x = wasserstein_distance(standardized_hist_x1.flatten(), standardized_hist_x2.flatten())
        # emd_y = wasserstein_distance(standardized_hist_y1.flatten(), standardized_hist_y2.flatten())
        # emd_x = kl_div(standardized_hist_x1, standardized_hist_x2).sum()
        # emd_y = kl_div(standardized_hist_y1, standardized_hist_y2).sum()
        emd_x = kl_div(hist_x1, hist_x2).sum()
        emd_y = kl_div(hist_y1, hist_y2).sum()

        return emd_x, emd_y

    def do_nms(self, threshold):
        self.all_bbox = nms_multiclass(self.all_bbox, threshold)

    def write_string(self):
        if self.language == 'en':
            with open(self.out_path + 'english_predicted.txt', 'w') as file:
                file.write(self.string)
        else:
            with open(self.out_path + 'chinese_predicted.txt', 'w') as file:
                file.write(self.string)

    def run_recognize(self):
        # EN
        if self.language == 'en':
            for character in self.character_params.keys():
                self.current = dict()
                self.current['character'] = character
                raw_template = cv2.imread(self.template_path + f'typed/{character}.png', cv2.IMREAD_GRAYSCALE)
                self.current['template'] = raw_template
                self.preprocess_template()
                cv2.imwrite(self.template_path + f'typed_adjusted/{character}.png', self.current['template'])
                self.template_match(input_image=self.preprocessed_image, candidate_bound=(1, 10), update=True)
                # self.visual_match_result()
            self.recognize(full=True)
            self.recognize(full=False)
            print('Recognize Result:')
            print(self.string)
            self.write_string()
        # CH
        else:
            file_list = sorted(os.listdir(self.words_path),
                               key=lambda x: tuple(map(int, x.replace('word', '').replace('.png', '').split('_'))))
            self.string = ''

            self.typed_templates = {}
            for hanzi in self.character_params.keys():
                raw_template = cv2.imdecode(np.fromfile(self.template_path + f'typed/{hanzi}.png',
                                                        dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                self.typed_templates[hanzi] = raw_template

            prev_x = None
            for name in tqdm.tqdm(file_list):
                x, y = map(int, name.replace('word', '').replace('.png', '').split('_'))
                if prev_x is not None and prev_x != x:
                    self.string += ''     # \n
                prev_x = x
                hanzi_image = cv2.imread(self.words_path + name, cv2.IMREAD_GRAYSCALE)

                peaks = {}
                hanzi_image = cv2.GaussianBlur(hanzi_image, ksize=(11, 11), sigmaX=2)
                for hanzi in self.typed_templates:
                    self.current['character'] = hanzi
                    self.current['template'] = self.typed_templates[hanzi]
                    self.current['template'] = cv2.GaussianBlur(self.current['template'], ksize=(11, 11), sigmaX=2)
                    self.template_match(input_image=hanzi_image, candidate_bound=(10, 20), threshold_rate=0.7)
                    peaks[hanzi] = np.max(self.current['density_map'])
                    # pad = (template_h - template_w) // 2
                    # self.current['template'] = np.pad(raw_template, ((0, 0), (pad, pad)), 'edge')
                    # cv2.imencode('.png', self.current['template'])[1].tofile(self.template_path + f'typed_adjusted/{hanzi}.png')
                self.string += max(peaks, key=peaks.get)
            print('Recognize Result:')
            print(self.string)
            self.write_string()



