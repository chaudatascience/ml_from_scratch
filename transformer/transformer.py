## Pytorch Implementation of Transformer model
# Paper "Attention Is All You Need" -  https://arxiv.org/abs/1706.03762

import copy
import math

import torch
import torch.nn as nn


def get_clones(module, num_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_state_dict_shapes_and_names(model):
    print(model.state_dict().keys())
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Some params are not trainable!')


class MultiHeadAttention(nn.Module):
    """
        multi-head self-attention:
    """

    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "emb_dim should be divisible by num_heads!"
        self.head_dim = model_dim // num_heads
        self.model_dim = model_dim
        self.num_heads = model_dim

        # In the paper, output of Q and K have d_k dimensions, while V's is d_v.
        # These values can be different, but they choose d_k = d_v = head_dim though.
        self.d_k, self.d_v = self.head_dim, self.head_dim

        # Linear Projections for Q, K, V
        self.QKV = nn.ModuleList([nn.Linear(self.model_dim, self.d_k * self.num_heads),
                                  nn.Linear(self.model_dim, self.d_k * self.num_heads),
                                  nn.Linear(self.model_dim, self.d_v * self.num_heads)])

        self.softmax = nn.Softmax(dim=-1)

        # Fully connected layer at the end with a shape of (num_heads*d_v, model_dim)
        self.W = nn.Linear(self.num_heads * self.d_v, model_dim)

    def forward(self, query, key, value, mask):
        """
        query: [batch_size, num_query_tokens, model_dim], in short: (b, q, model_dim)
        key: [batch_size, num_key_tokens, model_dim],
        value: [batch_size, num_value_tokens, model_dim]
        Note: num_key_tokens is equal to num_value_tokens, i.e., each token key corresponds to a token value.
        """

        batch_size = query.shape[0]

        ## Follow steps of Fig 2 (page 4) in the paper
        # Feed query, key, value into linear layers:
        query, key, value = [linear(x) for linear, x in zip(self.QKV, [query, key, value])]
        query = query.reshape(batch_size, -1, self.num_heads, self.d_k)
        key = key.reshape(batch_size, -1, self.num_heads, self.d_k)
        value = value.reshape(batch_size, -1, self.num_heads, self.d_v)  # d_v

        ## Dot product of each pair (query, key):
        # q, k are num_query_tokens, num_key_tokens respectively
        similarity = torch.einsum("bqhd,bkhd->bhqk", query, key)

        ## scale
        similarity /= (self.d_k ** .5)

        ## mask (optional): use to mask pad tokens and future tokens
        if mask is not None:  ## TODO: check on this
            similarity = similarity.masked_fill(mask == torch.tensor(False), float("-inf"))

        ## Softmax
        attention_weights = self.softmax(similarity)  # sum of weights for each query = 1

        ## Compute weighted sum of values:
        # attention_weights: (batch_size, num_heads, num_query_tokens, num_tokens)
        # value: (batch_size, num_tokens, num_heads, d_v)
        # Note: num_tokens here means num_value_tokens = num_key_tokens = num_tokens
        attention = torch.einsum("bhqt,bthv->bqhv", attention_weights, value)
        attention = attention.reshape(batch_size, -1, self.num_heads * self.d_v)  # (b, q, h*v)

        out = self.W(attention)  # out: (batch_size, num_query_tokens, model_dim)
        return out


class FeedForwardNet(nn.Module):
    """
    FFN(x) = max(0, x*W1 + b1)*W2 + b2 (page 5, eq. 2).
    In the paper, model_dim = 512, hid_dim = 2048.
    dropout: optional
    ## TODO: Why the paper sets hid_dim = 4 * model_dim?
    """

    def __init__(self, model_dim, hid_dim, dropout=0):
        super().__init__()
        self.linear_1 = nn.Linear(model_dim, hid_dim)
        self.linear_2 = nn.Linear(hid_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # optional

    def forward(self, x):
        """
        x: (batch_size, num_query_tokens, model_dim)
        """
        out = self.relu(self.linear_1(x))
        out = self.dropout(out)  # optional, not in the paper
        out = self.linear_2(out)
        return out


class EncoderLayer(nn.Module):
    """
    Consisting of two sub-layers:
        + multi-head self-attention mechanism
        + position-wise fully connected feed-forward network.
    Output of each sub-layer is LayerNorm(x+ Sublayer(x))
    Note: We can make it more general by defining a SubLayer class. In this way,
    an EncoderLayer can consist of many sub-layers rather than 2.
    """

    def __init__(self, model_dim, num_heads, hid_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads)
        self.ffnet = FeedForwardNet(model_dim, hid_dim, dropout)
        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)  # TODO: Should we re-use `norm_1`

        self.dropout = nn.Dropout(dropout)  # optional

    ## We can pass 2 params: x and mask, it works just fine for the encoder.
    # But here, we're passing query, key, value to reuse this class in DecoderLayer.
    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)
        out = self.norm_1(query + attention)
        out = self.dropout(out)  # optional, not in the paper

        out_2 = self.ffnet(out)
        out_2 = self.norm_2(out + out_2)
        out_2 = self.dropout(out_2)  # optional, not in the paper
        return out_2


class Encoder(nn.Module):
    def __init__(self, num_encoder_layers, model_dim, num_heads, hid_dim, dropout):
        super().__init__()
        self.encoder = get_clones(EncoderLayer(model_dim, num_heads, hid_dim, dropout), num_encoder_layers)
        self.norm = nn.LayerNorm(model_dim)  # optional

    def forward(self, x, mask):
        for encoder_layer in self.encoder:
            x = encoder_layer(x, x, x, mask)
        x = self.norm(x)  # optional
        return x


class DecoderLayer(nn.Module):
    """
    We can re-use EncoderLayer based on an observation that the DecoderLayer has 3 sub-layers,
    in which 2 of them are the same as the EncoderLayer.
    """

    def __init__(self, model_dim, num_heads, hid_dim, dropout):
        super().__init__()

        self.masked_attention = MultiHeadAttention(model_dim, num_heads)
        self.norm = nn.LayerNorm(model_dim)
        self.attention_and_ffn = EncoderLayer(model_dim, num_heads, hid_dim, dropout)

    def forward(self, encoder_key, encoder_value, x, src_mask, trg_mask):
        attention = self.masked_attention(x, x, x, trg_mask)
        decoder_query = self.norm(x + attention)
        out = self.attention_and_ffn(query=decoder_query, key=encoder_key, value=encoder_value, mask=src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, num_decoder_layers, model_dim, num_heads, hid_dim, dropout):
        super().__init__()
        self.decoder = get_clones(DecoderLayer(model_dim, num_heads, hid_dim, dropout), num_decoder_layers)
        self.norm = nn.LayerNorm(model_dim)  # optional

    def forward(self, encoder_key, encoder_value, x, src_mask, trg_mask):
        for decoder_layer in self.decoder:
            x = decoder_layer(encoder_key, encoder_value, x, src_mask, trg_mask)
        out = self.norm(x)  # optional
        return out


class PositionalEncoding(nn.Module):
    # src: https://github.com/gordicaleksa/pytorch-original-transformer/
    def __init__(self, model_dim, dropout, max_sequence_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position_id = torch.arange(0, max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dim, 2, dtype=torch.float) / model_dim)

        positional_encodings_table = torch.zeros(max_sequence_length, model_dim)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions

        # Register buffer to the state_dict.
        # Side note: Buffers won't be returned in model.parameters(),
        # thus the optimizer won't update them.
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        # broadcast and add it to embeddings
        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        # Page 7, Section 5.4 "Regularization"
        return self.dropout(embeddings_batch + positional_encodings)


class OutputDecoding(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super().__init__()
        self.linear_layer = nn.Linear(model_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(-1)

    def forward(self, x):
        return self.log_softmax(self.linear_layer(x))


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size,
                 num_encoder_layers, model_dim, num_heads, hid_dim, dropout, num_decoder_layers,
                 ):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, model_dim)
        self.trg_emd = nn.Embedding(trg_vocab_size, model_dim)

        self.src_positional_encoding = PositionalEncoding(model_dim, dropout)
        self.trg_positional_encoding = PositionalEncoding(model_dim, dropout)

        if hid_dim is None:
            hid_dim = model_dim * 4
        self.encoder = Encoder(num_encoder_layers, model_dim, num_heads, hid_dim, dropout)
        self.decoder = Decoder(num_decoder_layers, model_dim, num_heads, hid_dim, dropout)

        self.linear = OutputDecoding(model_dim, trg_vocab_size)

        self.model_dim = model_dim

        self.init_params(use_xavier_init = True)

    def init_params(self, use_xavier_init):
        if use_xavier_init:
            for name, param in self.named_parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)


        ## TODO: check on this
        # Note: the vocab size should be the same.
        # share weight between emb layers and logit layer (mentioned in the paper)
        self.trg_emd.weight = self.src_emb.weight ## point to the same parameter (tensor)
        self.linear.linear_layer.weight = self.src_emb.weight

        nn.init.normal_(self.src_emb.weight, mean=0., std=math.pow(self.model_dim, -0.5))



    def encode(self, src_token_ids, src_mask):
        x = self.src_emb(src_token_ids)
        x = self.src_positional_encoding(x)
        output = self.encoder(x, src_mask)
        return output

    def decode(self, encoder_out, trg_token_ids, src_mask, trg_mask):
        x = self.trg_emd(trg_token_ids)
        x = self.trg_positional_encoding(x)
        trg_representations = self.decoder(encoder_out, encoder_out, x, src_mask, trg_mask)

        log_probs = self.linear(trg_representations)  # log_probabilities: (b, t, trg_vocab_size)
        log_probs = log_probs.reshape(-1, log_probs.shape[-1])  # TODO: check on this
        return log_probs

    def forward(self, src_token_ids, trg_token_ids, src_mask, trg_mask):
        src_representations = self.encode(src_token_ids, src_mask)
        trg_log_probs = self.decode(src_representations, trg_token_ids, src_mask, trg_mask)
        return trg_log_probs


if __name__ == "__main__":

    src_vocab_size = 15
    trg_vocab_size = 15
    src_token_ids_batch = torch.randint(5, 10, size=(3, 2))
    trg_token_ids_batch = torch.randint(5, 10, size=(3, 2))
    model_dim = 512

    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        num_encoder_layers=2,
        num_decoder_layers=2,
        model_dim=model_dim,
        hid_dim=model_dim*4,
        num_heads=2,
        dropout=0.1
    )

    analyze_state_dict_shapes_and_names(transformer)
    print(f'Num of parameters: {count_parameters(transformer)}')

    output = transformer(src_token_ids_batch, trg_token_ids_batch, src_mask=None, trg_mask=None)

