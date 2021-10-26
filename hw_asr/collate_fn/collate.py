import logging
from typing import List
import torch
from torch import nn
logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}
    result_batch['spectrogram'] = []
    result_batch['spectrogram_length'] = []
    result_batch['text_encoded'] = []
    result_batch['text_encoded_length'] = []
    result_batch['text'] = []
    # TODO: your code here
    for i in range(len(dataset_items)):
        # .squeeze(0) : [1, 128, 116] -> [128, 116]
        # .transpose(1, 0) : [128, 116] -> [116, 128]
        result_batch['spectrogram'].append(dataset_items[i]['spectrogram'].squeeze(0).transpose(1, 0))
        # shape[2] from [1, 128, 116]
        result_batch['spectrogram_length'].append(dataset_items[i]['spectrogram'].shape[2])
        # .squeeze(0) : [1, 3] -> [3]
        result_batch['text_encoded'].append(dataset_items[i]['text_encoded'].squeeze(0))
        result_batch['text_encoded_length'].append(dataset_items[i]['text_encoded'].shape[1])
        result_batch['text'].append(dataset_items[i]['text'])
    result_batch['spectrogram'] = nn.utils.rnn.pad_sequence(result_batch['spectrogram'], batch_first=True)
    result_batch['text_encoded'] = nn.utils.rnn.pad_sequence(result_batch['text_encoded'], batch_first=True)
    result_batch['text_encoded_length'] = torch.tensor(result_batch['text_encoded_length'], dtype=torch.long)
    result_batch['spectrogram_length'] = torch.tensor(result_batch['spectrogram_length'], dtype=torch.long)
    return result_batch
