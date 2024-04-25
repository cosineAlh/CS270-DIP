import Levenshtein
import numpy as np
import matplotlib.pyplot as plt


# ENGLISH_GROUND_TRUTH_WITH_PUNCTUATION = "Right now , there are people all over the world who are just like you .\n" +\
#                                         "They're lonely . They're missing somebody . They're in love with someone\n" +\
#                                         "they probably shouldn't be in love with . They have secrets you wouldn't\n" +\
#                                         "believe . They wish , dream , hope , and they look out the window whenever\n" +\
#                                         "they're in the car or on a bus or a train & they watch the people on the\n" +\
#                                         "streets & wonder what they've been through . They wonder if there are p-\n" +\
#                                         "eople out there like them . They're like you & you could tell them everyth-\n" +\
#                                         "ing & they would understand . You're never alone ."

ENGLISH_GROUND_TRUTH_WITHOUT_PUNCTUATION = "Right now there are people all over the world who are just like you " +\
                                           "Theyre lonely Theyre missing somebody Theyre in love with someone " +\
                                           "they probably shouldnt be in love with They have secrets you wouldnt " +\
                                           "believe They wish dream hope and they look out the window whenever " +\
                                           "theyre in the car or on a bus or a train they watch the people on the " +\
                                           "streets wonder what theyve been through They wonder if there are p " +\
                                           "eople out there like them Theyre like you you could tell them everyth " +\
                                           "ing they would understand Youre never alone"

CHINESE_GROUND_TRUTH_WITHOUT_PUNCTUATION = "芙蓉楼送辛渐" +\
                                           "唐代王昌龄" +\
                                           "寒雨连江夜入吴" +\
                                           "平明送客楚山孤" + \
                                           "洛阳亲友如相问" + \
                                           "一片冰心在玉壶"

ENGLISH_GROUND_TRUTH_STATS = {'e': 77, 'o': 38, 'h': 37, 't': 34, 'r': 29, 'n': 23, 'l': 23, 'y': 21, 'i': 17, 'a': 16,
                              'w': 15, 'u': 15, 'd': 13, 's': 13, 'v': 9, 'p': 8, '\n': 7, 'T': 7, 'b': 7, 'm': 6, 'g': 4,
                              'k': 4, 'c': 4, 'R': 1, 'j': 1, 'f': 1, 'Y': 1}


def calculate_wer(predicted, ground_truth):
    predicted_words = predicted.split(' ')
    ground_truth_words = ground_truth.split(' ')
    distance = Levenshtein.distance(predicted_words, ground_truth_words)
    wer = distance / len(ground_truth_words) * 100
    return wer


def calculate_cer(predicted, ground_truth):
    distance = Levenshtein.distance(predicted, ground_truth)
    cer = distance / len(ground_truth) * 100
    return cer


def calculate_word_accuracy(predicted, ground_truth, language):
    if language == 'en':
        predicted_words = predicted.split()
        ground_truth_words = ground_truth.split()
    else:
        predicted_words = list(predicted)
        ground_truth_words = list(ground_truth)
    total_words = len(ground_truth_words)
    correct_words = sum(1 for pred, gt in zip(predicted_words, ground_truth_words) if pred == gt)
    accuracy = correct_words / total_words * 100
    return accuracy


def find_wrong_word_pairs(predicted, ground_truth, language):
    out_str = 'Wrong Pairs (GT -> Pred):\n'
    if language == 'en':
        predicted_words = predicted.split()
        ground_truth_words = ground_truth.split()
    else:
        predicted_words = list(predicted)
        ground_truth_words = list(ground_truth)

    # Iterate over the words and compare
    for pred_word, gt_word in zip(predicted_words, ground_truth_words):
        if pred_word != gt_word:
            out_str += f'{gt_word} -> {pred_word}\n'

    return out_str


def evaluate_english(predicted, out_path):
    report_string = ''

    WER = calculate_wer(predicted, ENGLISH_GROUND_TRUTH_WITHOUT_PUNCTUATION)
    report_string += f'WER: {WER}\n'
    CER = calculate_cer(predicted, ENGLISH_GROUND_TRUTH_WITHOUT_PUNCTUATION)
    report_string += f'CER: {CER}\n'
    WACC = calculate_word_accuracy(predicted, ENGLISH_GROUND_TRUTH_WITHOUT_PUNCTUATION, 'en')
    report_string += f'WACC: {WACC}\n'
    WWP = find_wrong_word_pairs(predicted, ENGLISH_GROUND_TRUTH_WITHOUT_PUNCTUATION, 'en')
    report_string += WWP

    with open(out_path + 'english_report.txt', 'w') as file:
        file.write(report_string)
    print('evaluation of English predicted:')
    print(report_string)


def evaluate_chinese(predicted, out_path):
    report_string = ''

    WACC = calculate_word_accuracy(predicted, CHINESE_GROUND_TRUTH_WITHOUT_PUNCTUATION, 'ch')
    report_string += f'WACC: {WACC}\n'
    WWP = find_wrong_word_pairs(predicted, CHINESE_GROUND_TRUTH_WITHOUT_PUNCTUATION, 'ch')
    report_string += WWP

    with open(out_path + 'chinese_report.txt', 'w') as file:
        file.write(report_string)
    print('evaluation of Chinese predicted:')
    print(report_string)


