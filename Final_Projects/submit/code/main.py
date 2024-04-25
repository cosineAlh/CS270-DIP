import cv2
from shear import shear
from denoise import poetry_denoise, handwritten_denoise
from segment import segment
from recognize import Recognizer
from evaluation import evaluate_english, evaluate_chinese


def English():
    """
    English handwritten
    """
    original_english = cv2.imread('../data/original/handwritten_English.jpg', cv2.IMREAD_COLOR)
    out_path = '../result/'
    temp_path = '../data/temp/'
    template_path = '../data/templates/'
    # # step 1
    # shear(original_english, temp_path)
    # # step 2
    sheared = cv2.imread(temp_path + 'sheared.png', cv2.IMREAD_GRAYSCALE)
    handwritten_denoise(sheared, temp_path)
    # # step 3, segment只选一个
    denoised = cv2.imread(temp_path + 'english_binary.png', cv2.IMREAD_GRAYSCALE)
    segment(sheared, sheared, temp_path + './lines/', temp_path + './words/', 'en')       # with-underline
    # segment(sheared, denoised, temp_path + './lines/', temp_path + './words/', 'en')        # without-underline
    # # step 4
    words_path = temp_path + 'words/'
    recognizer = Recognizer(sheared, template_path, temp_path, words_path, out_path, language='en')
    recognizer.run_recognize()
    # # step 5
    with open(out_path + 'english_predicted.txt', 'r') as file:
        string_predicted = file.read()
    evaluate_english(string_predicted, out_path)


def Chinese():
    """
    Chinese handwritten
    """
    original_chinese = cv2.imread('../data/original/poetry_Chinese.jpg', cv2.IMREAD_COLOR)
    out_path = '../result/'
    temp_path = '../data/temp/'
    template_path = '../data/templates/'
    # # step 1
    poetry_denoise(original_chinese, temp_path)
    # # step 2
    denoised = cv2.imread(temp_path + 'chinese_binary.png', cv2.IMREAD_GRAYSCALE)
    segment(denoised, denoised, temp_path + './hang/', temp_path + './hanzi/', 'ch')
    # # step 3
    words_path = temp_path + 'hanzi/'
    recognizer = Recognizer(denoised, template_path, temp_path, words_path, out_path, language='ch')
    recognizer.run_recognize()
    # # step 4
    with open(out_path + 'chinese_predicted.txt', 'r') as file:
        string_predicted = file.read()
    evaluate_chinese(string_predicted, out_path)


if __name__ == '__main__':
    English()
    Chinese()


