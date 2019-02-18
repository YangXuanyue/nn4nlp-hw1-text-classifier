import configs
import os
import numpy as np
import json
from collections import Counter, defaultdict


class Vocab:
    path = f'{configs.data_dir}/vocab.txt'
    embedding_mat_path = f'{configs.data_dir}/embedding_mat.fasttext-wiki-word.npy'

    @staticmethod
    def build(words):
        word_to_id, id_to_word = {}, []

        # for word in ('<pad>', '<s>', '</s>', '<unk>'):
        for word in ('<pad>', '<unk>'):
            word_to_id[word] = len(id_to_word)
            id_to_word.append(word)

        word_cnts = Counter(words)

        # with open(f'{configs.data_dir}/word_cnts.json', 'w') as word_cnts_file:
        #     json.dump(word_cnts, word_cnts_file)

        for word, cnt in word_cnts.items():
            if cnt > 5:
                word_to_id[word] = len(id_to_word)
                id_to_word.append(word)

        with open(Vocab.path, 'w') as vocab_file:
            vocab_file.writelines('\n'.join(id_to_word))

        return Vocab()

    def __init__(self):
        assert os.path.exists(Vocab.path)

        self.word_to_id, self.id_to_word = defaultdict(lambda: self.unk_id), []

        with open(Vocab.path) as vocab_file:
            for word in map(lambda s: s.strip(), vocab_file.readlines()):
                self.word_to_id[word] = len(self.id_to_word)
                self.id_to_word.append(word)

        self.padding_id = self.word_to_id['<pad>']
        # self.start_id = self.word_to_id['<s>']
        # self.end_id = self.word_to_id['</s>']
        self.unk_id = self.word_to_id['<unk>']
        self.size = len(self.id_to_word)
        assert len(self.word_to_id) == self.size

    def build_embedding_mat(self, new=False):
        if new or not os.path.exists(Vocab.embedding_mat_path):
            embedding_mat = np.random.randn(self.size, configs.word_embedding_dim).astype(np.float32)
            embedding_mat[self.padding_id].fill(0.)
            assert len(self.word_to_id) == self.size

            print(embedding_mat.shape)

            with open(configs.word_embeddings_path) as word_embeddings_file:
                line = word_embeddings_file.readline()
                embedding_num, embedding_dim = map(int, line.strip().split())
                line_cnt = 0
                overlap_cnt = 0

                assert embedding_dim == configs.word_embedding_dim

                words = set()

                for line in word_embeddings_file.readlines():
                    word, *embedding = line.split()
                    line_cnt += 1
                    assert len(embedding) == embedding_dim
                    words.add(word)

                    if word in self.word_to_id:
                        overlap_cnt += 1
                        assert self.word_to_id[word] is not self.unk_id
                        embedding = np.array(
                            list(map(float, embedding))
                        )

                        np.copyto(
                            dst=embedding_mat[self.word_to_id[word]],
                            src=list(map(float, embedding))
                        )
                        assert np.allclose(embedding_mat[self.word_to_id[word]], np.array(list(map(float, embedding))))

                assert len(words) == embedding_num
                print(len(words & set(self.id_to_word)))
                print(sum(word in self.word_to_id for word in words))
                assert line_cnt == embedding_num

            np.save(Vocab.embedding_mat_path, embedding_mat)

            print(f'built embedding matrix from {configs.word_embeddings_path} '
                  f'with {overlap_cnt} overlaps / {self.size}')
        else:
            embedding_mat = np.load(Vocab.embedding_mat_path)
            print(f'loaded embedding matrix from {configs.word_embeddings_path}')

        return embedding_mat


    # def __contains__(self, word):
    #     return word in self.word_to_id

    def __getitem__(self, word_or_id):
        if isinstance(word_or_id, str):
            return self.word_to_id[word_or_id]
        else:
            return self.id_to_word[word_or_id]

    def textify(self, ids):
        return ' '.join(map(self.id_to_word.__getitem__, ids))

    def idify(self, words):
        # return list(map(self.word_to_id.__getitem__, words))
        return list(map(lambda word: self.word_to_id[word.lower()], words))
