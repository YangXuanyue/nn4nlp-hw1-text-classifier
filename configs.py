import datetime
import argparse as ap
from model_utils import *
import time
import random
import os


arg_parser = ap.ArgumentParser()
arg_parser.add_argument('-m', '--mode', default='train')
arg_parser.add_argument('-c', '--ckpt', default=None)
# arg_parser.add_argument('-d', '--device', type=int, default=0)
arg_parser.add_argument('-b', action='store_true', default=False)
arg_parser.add_argument('-l', action='store_true', default=False)
arg_parser.add_argument('-s', '--seed', type=int, default=None)
arg_parser.add_argument('-wd', '--weight_decay', type=float, default=0)

args = arg_parser.parse_args()

mode = args.mode
training = args.mode == 'train'
validating = args.mode == 'dev'
testing = args.mode == 'test'
# ckpt.best is trained with data_part_num = 3
ckpt_id = args.ckpt
# device_id = args.device
loads_best_ckpt = args.b
loads_ckpt = args.l


seed = args.seed if args.seed is not None else int(time.time())
print(f'seed={seed}')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# training = True
# ckpt_id = None
# loads_best_ckpt = False
# loads_ckpt = False


timestamp = datetime.datetime.now().strftime('%m%d-%H%M%S')


class Dir:
    def __init__(self, name):
        self.name = name

        if not os.path.exists(name):
            os.mkdir(name)

    def __str__(self):
        return self.name


logs_dir = Dir('logs')
ckpts_dir = Dir('ckpts')
data_dir = Dir('topicclass')
# lm_ckpts_dir = 'lm_ckpts'

best_ckpt_path = f'{ckpts_dir}/ckpt.best'

# word_embeddings_files
# ├── fasttext
# │   ├── crawl-300d-2M-subword.vec
# │   ├── crawl-300d-2M.vec
# │   ├── wiki-news-300d-1M-subword.vec
# │   └── wiki-news-300d-1M.vec
# ├── glove
# │   ├── glove.42B.300d.txt
# │   ├── glove.6B.100d.txt
# │   ├── glove.6B.200d.txt
# │   ├── glove.6B.300d.txt
# │   ├── glove.6B.50d.txt
# │   ├── glove.840B.300d.txt
# │   ├── glove.twitter.27B.100d.txt
# │   ├── glove.twitter.27B.200d.txt
# │   ├── glove.twitter.27B.25d.txt
# │   └── glove.twitter.27B.50d.txt
# └── word2vec
#     └── GoogleNews-vectors-negative300.bin

word_embeddings_path = 'word_embeddings_files/' \
                       'fasttext/' \
                       'wiki-news-300d-1M.vec'
word_embedding_dim = 300
inits_embedder = True
freezes_embeddings = True  # False
embedder_training_epoch_num = 12
uses_new_embeddings = False

kernel_widths = [3, 4, 5]
kernel_nums = [100, 100, 100]
# kernel_nums = [100, 100, 100]
feature_num = sum(kernel_nums)

hidden_size = 256

key_size, value_size, query_size = [256] * 3
# key_size, value_size, query_size = [128] * 3

classes_path = 'classes.txt'
class_num = -1 # to be changed

adam_lr = 1e-4
adadelta_lr = 1e-2
# lm_lr = 5e-5
sets_new_lr = False
batch_size = 256
uses_bi_rnn = True
rnn_type = 'lstm'
# rnn_type = 'gru'
rnn_hidden_size = 512
rnn_layer_num = 4
min_logit = -1e3
dropout_prob = .5
momentum = .9
l2_weight_decay = 5e-4 # args.weight_decay
# feature_num = 512
max_acc_path = 'max_acc.txt'

epoch_num = 12

# uses_center_loss = False
# uses_pairwise_sim_loss = False

normalizes_outputs = True
uses_new_classifier = False
uses_new_transformer = False
uses_new_encoder = False

uses_cnn = True

uses_batch_norm_encoder = False
uses_residual_rnns = False

loads_optimizer_state = not (sets_new_lr or uses_new_classifier or uses_new_encoder)

lr_scheduler_factor = .5
lr_scheduler_patience = 100

uses_weight_dropped_rnn = True

rnn_weights_dropout_prob = 0 # .2

tests_ensemble = False

sampling_rate = .1  # .1

encoder_device_id = 0
decoder_device_id = 1

per_node_beam_width = 32
beam_width = 32

uses_new_optimizer = False

uses_self_attention = True

uses_gumbel_softmax = False

uses_lang_model = False

use_multi_head_attention = False

# uses_lm = not training and False
