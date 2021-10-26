# torchaudio.transforms.FrequencyMasking()
# torchaudio.transforms.TimeMasking()
#
# torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
# torchaudio.transforms.TimeMasking(time_mask_param=35)
# import torch_audiomentations
# from torch import Tensor
# from hw_asr.augmentations.base import AugmentationBase
# import torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
import random
import torchaudio
from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase

class Augmentations(AugmentationBase):
    # augmentaton_type = ['Frequency', 'Time', 'Mixed']
    def __init__(self, *args, **kwargs):
        self._aug_freq = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self._aug_time = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        cur_type = random.randint(0, 2)
        if cur_type == 0:
            return self._aug_freq(x).squeeze(1)
        elif type == 1:
            return self._aug_time(x).squeeze(1)
        else:
            return self._aug_time(self._aug_freq(x)).squeeze(1)

# class FrequencyMasking(AugmentationBase):
#     def __init__(self, *args, **kwargs):
#         self._aug = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
#
#     def __call__(self, data: Tensor):
#         x = data.unsqueeze(1)
#         return self._aug(x).squeeze(1)
#
# class TimeMasking(AugmentationBase):
#     def __init__(self, *args, **kwargs):
#         self._aug = torchaudio.transforms.TimeMasking(time_mask_param=35)
#
#     def __call__(self, data: Tensor):
#         x = data.unsqueeze(1)
#         return self._aug(x).squeeze(1)


# from hw_asr.augmentations.base import AugmentationBase
#
#
# class Gain(AugmentationBase):
#     def __init__(self, *args, **kwargs):
#         self._aug = torch_audiomentations.Gain(*args, **kwargs)
#
#     def __call__(self, data: Tensor):
#         x = data.unsqueeze(1)
#         return self._aug(x).squeeze(1)