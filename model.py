import configs
from model_utils import *
from modules import *
import data_utils


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # self.embedder = nn.Embedding(
        #     num_embeddings=data_utils.vocab.size,
        #     embedding_dim=configs.word_embedding_dim,
        #     padding_idx=data_utils.vocab.padding_id,
        #     _weight=(
        #         torch.from_numpy(data_utils.vocab.build_embedding_mat())
        #         if configs.inits_embedder else None
        #     )
        # ).cuda()
        self.embedder = nn.Embedding.from_pretrained(
            embeddings=torch.from_numpy(
                data_utils.vocab.build_embedding_mat(new=configs.uses_new_embeddings)
            ), freeze=configs.freezes_embeddings
        ).cuda()

        if not configs.freezes_embeddings:
            with torch.no_grad():
                self.embedder.weight[data_utils.vocab.padding_id].requires_grad_(False)

        # if configs.fixes_embeddings:
        #     self.embedder.weight.requires_grad_(False)

        self.feature_extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=configs.word_embedding_dim,
                        out_channels=kernel_num,
                        kernel_size=kernel_width,
                        # padding=1
                        # stride=0, padding=1, dilation=0
                    ),
                    # nn.BatchNorm1d(num_features=kernel_num),
                    # nn.ReLU(),
                    nn.Tanh(),
                    # nn.Dropout(configs.dropout_prob),
                    # nn.Conv1d(
                    #     in_channels=kernel_num,
                    #     out_channels=kernel_num,
                    #     kernel_size=3,
                    #     stride=1, padding=0, dilation=1
                    # ),
                    # nn.BatchNorm1d(num_features=kernel_num),
                    # nn.ReLU(),
                    nn.AdaptiveMaxPool1d(output_size=1),
                    Reshaper(-1, kernel_num)
                )
                for kernel_width, kernel_num in zip(configs.kernel_widths, configs.kernel_nums)
            ]
        ).cuda()
        # self.classifier = nn.Linear(configs.feature_num, configs.class_num).cuda()
        self.classifier = nn.Sequential(
            # nn.Linear(configs.feature_num, configs.hidden_size  // 2),
            # nn.BatchNorm1d(num_features=configs.hidden_size // 2),
            # nn.ReLU(),
            nn.Dropout(configs.dropout_prob),
            # nn.Linear(configs.hidden_size, configs.hidden_size // 2),
            # nn.BatchNorm1d(num_features=(configs.hidden_size // 2)),
            # nn.ReLU(),
            # nn.Dropout(configs.dropout_prob),
            nn.Linear(configs.feature_num, configs.class_num)
        ).cuda()

        self.init_weights()

    def init_weights(self):
        self.apply(init_weights)

    def forward(
            self,
            # [batch_size, text_len]
            text_batch,
    ):
        # [batch_size, embedding_dim, text_len]
        embeddings_batch = self.embedder(text_batch).transpose_(1, 2)

        # print(embeddings_batch.shape)

        # [batch_size, feature_num]
        feature_vec_batch = torch.cat(
            [
                # [batch_size, kernel_num]
                feature_extractor(embeddings_batch)
                for feature_extractor in self.feature_extractors
            ], dim=-1
        )
        # feature_vec_batch = F.dropout(feature_vec_batch, p=configs.dropout_prob, training=self.training)
        # [batch_size, class_num]
        return self.classifier(feature_vec_batch)


