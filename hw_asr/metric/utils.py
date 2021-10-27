# Don't forget to support cases when target_text == ''
import Levenshtein
from jiwer import wer
import re

def calc_cer(target_text, predicted_text) -> float:
    # TODO: your code here
    # Calculate the Levenshtein distance between predictions and GT
    # CER = Levenshtein distance
#     print( 'CLEAR', re.sub('▁', ' ',predicted_text.strip('▁')))
#     print('CER: ', min(1, Levenshtein.distance(target_text, re.sub('▁', ' ',predicted_text.strip('▁'))) / (len(predicted_text) + 10 ** (-7))))
    return min(1, Levenshtein.distance(target_text, re.sub('▁', ' ',predicted_text.strip('▁'))) / (len(predicted_text) + 10 ** (-7)))


def calc_wer(target_text, predicted_text) -> float:
    # if predicted_text =="", WER is 1.0
    # print(target_text, predicted_text)
#     print('CLEAR', re.sub('▁', ' ',predicted_text.strip('▁')))
#     print('WER: ', wer(target_text, re.sub('▁', ' ',predicted_text.strip('▁'))))
    return wer(target_text, re.sub('▁', ' ',predicted_text.strip('▁')))
