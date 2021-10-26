import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union
import youtokentome as yttm
import numpy as np
from torch import Tensor
import re

# переопределеить через бпе енкод!
# и то же самое с декодом
class BPETextEncoder:
    def __init__(self):
        self.bpe = yttm.BPE(model='texts_str.model')

    def encode(self, text) -> Tensor:
        """encodes text to ints"""
        print('TEXT:  ', text)
        text = self.normalize_text(text)
        try:
            return Tensor(self.bpe.encode([text], output_type=yttm.OutputType.ID)[0]).unsqueeze(0)
        except:
            print('problem')
        # bpe.encode(text)
        # except KeyError as e:
        #     unknown_chars = set([char for char in text if char not in self.char2ind])
        #     raise Exception(
        #         f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'")

    def decode(self, vector):
        # def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        "decodes ints to tex"
        print(vector.tolist())
        print('DECODED: ', self.bpe.decode(vector.tolist())[0])
        return self.bpe.decode(vector.tolist())[0]


    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        res = ''
        prev_token = '<PAD>'
        for i in range(len(inds)):
            c = self.bpe.id_to_subword(inds[i])
            if c == '<PAD>' and prev_token == c:
                continue
            if c != prev_token and c != '<PAD>':
                res += c
            prev_token = c
        print('RES', res)
        return res

    def __len__(self):
        return self.bpe.vocab_size()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text