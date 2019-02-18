import configs
import os

from tqdm import tqdm
from model_utils import *
from collections import defaultdict, Counter
from itertools import chain
import time
import json
# import Levenshtein
import csv
from vocab import Vocab
import bisect
# from PIL import Image
import pdb
# import torch.multiprocessing as mp

# mp.set_start_method('spawn')

vocab = Vocab()

id_to_class, class_to_id = [], {}

with open(configs.classes_path) as classes_file:
    for class_ in map(lambda s: s.strip(), classes_file.readlines()):
        class_to_id[class_] = len(id_to_class)
        id_to_class.append(class_)

configs.class_num = len(class_to_id)

names = ('train', 'valid') if configs.training else ('test',)


class Dataset(tud.Dataset):
    def __init__(self, name):
        self.name = name
        self.texts = np.load(f'{configs.data_dir}/texts.id.{name}.npy')
        self.labels = np.load(f'{configs.data_dir}/labels.{name}.npy')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], idx


datasets = {
    name: Dataset(name)
    for name in names
}


def get_dataset_size(name):
    return len(datasets[name])


def collate(batch):
    text_batch, label_batch, idx_batch = zip(*batch)

    # [batch_size]
    len_batch = torch.LongTensor(
        [len(text) for text in text_batch]
    )
    max_len = max(len_batch)
    # [batch_size, max_len]
    text_batch = torch.LongTensor(
        [
            np.concatenate((text, np.full((max_len - len(text)), vocab.padding_id)))
            if len(text) < max_len else text
            for text in text_batch
        ]
    )
    label_batch = torch.LongTensor(label_batch)

    return text_batch.cuda(), label_batch.cuda()



data_loaders = {
    name: tud.DataLoader(
        dataset=datasets[name],
        batch_size=configs.batch_size,
        shuffle=(name == 'train'),
        # pin_memory=True,
        collate_fn=collate,
        # num_workers=4
    )
    for name in names
}


def gen_batches(name):
    instance_num = 0

    # print(f'num_workers = {data_loaders[name].num_workers}')

    for batch in data_loaders[name]:
        instance_num += len(batch[-1])
        pct = instance_num * 100. / len(datasets[name])
        yield pct, batch # map(lambda b: b.cuda(), batch)


def save_predictions(name, predictions):
    if name == 'valid':
        results = []

        for prediction, word_ids, label in zip(predictions, datasets['valid'].texts, datasets['valid'].labels):
            results.append(
                {
                    'text': vocab.textify(word_ids),
                    'correct': bool(int(prediction) == label),
                    'label': id_to_class[label],
                    'prediction':  id_to_class[prediction]
                }
            )

        json.dump(results, open('results.json', 'w'), indent=4)
    else:
        np.save('predictions.npy', predictions)