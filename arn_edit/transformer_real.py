import torch
from torch import nn
import numpy as np 
import math


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_tokens=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.positional_encoding = torch.zeros((1, max_tokens, d_model))
        i = torch.arange(max_tokens, dtype=torch.float32).reshape(-1, 1)
        X = i / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        self.positional_encoding[:, :, 0::2] = torch.sin(X)
        self.positional_encoding[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.positional_encoding[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# 点积注意力机制
class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def masked_softmax(self, X, masked=None):
        if masked is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            # 添加Mask
            shape = X.shape # (Batch * N_head) * Length * Length
            masked = torch.repeat_interleave(masked, shape[1])  # 1D (N_head * Length * Length)
            mask = torch.arange(shape[1])[None,:]  < masked[:,None]
            mask = mask.reshape(shape)
            X[~mask] = -1e6
            return nn.functional.softmax(X, dim=-1) # (Batch * N_head) * Length * Length
    
    def forward(self, Q, K, V, valid_lens=None): # (Batch * N_head) * Length * num_hidden 
        d = Q.shape[-1] # num_hidden 
        scores = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(d) # (Batch * N_head) * Length * Length
        self.weights = self.masked_softmax(scores, valid_lens) # (Batch * N_head) * Length * Length
        return torch.bmm(self.dropout(self.weights), V) # (Batch * N_head) * Length * D_model

# 多头自注意力
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden=None, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.n_heads = n_heads   
        if d_hidden is None:
            d_hidden = d_model
        assert d_model % self.n_heads == 0
        # 线性变换层
        self.q = nn.Linear(d_model, d_hidden)
        self.k = nn.Linear(d_model, d_hidden)
        self.v = nn.Linear(d_model, d_hidden)
        self.attention = DotProductAttention(dropout)
        self.fc = nn.Linear(d_hidden, d_model)
        
    def transpose(self, X, n_heads):
        # X : Batch * Length * size
        X = X.reshape(X.shape[0], X.shape[1], n_heads, -1) # Batch * Length * N_head * size//N_head
        X = X.permute(0, 2, 1, 3) # Batch * N_head * Length * size//N_head
        return X.reshape(-1, X.shape[2], X.shape[3]) # (Batch * N_head) * Length * size//N_head

    def retranspose(self, X, n_heads):
        X = X.reshape(-1, n_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1) # Batch * Length * size

    def forward(self, Q, K, V, masked=None):
        Q = self.transpose(self.q(Q), self.n_heads) # (Batch * N_head) * Length * num_hidden
        K= self.transpose(self.k(K), self.n_heads)
        V = self.transpose(self.v(V), self.n_heads)
        
        if masked is not None:
            # 处理匹配多头
            masked = torch.repeat_interleave(masked, self.n_heads) # 1D (N_head * Length)
        
        output = self.attention(Q, K, V, masked) # (Batch * N_head) * Length * D_model
        output_concat = self.retranspose(output, self.n_heads)
        return self.fc(output_concat)

# 前馈神经网络
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = n.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 残差&Norm
class AddNorm(nn.Module):
    def __init__(self, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, y):
        x = self.dropout(x + y)
        layer_norm = nn.LayerNorm(x.shape[1:])
        return layer_norm(x)

# 编码器Transformer块
class EncoderTransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, hidden_dim=2048, attention_dropout=0.1, addnorm_dropout=0.1):
        super(EncoderTransformerBlock, self).__init__()
        # self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=pos_dropout, max_tokens=max_token)
        self.attention = MultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=attention_dropout)
        self.addnorm1 = AddNorm(dropout=addnorm_dropout)
        self.ffn = FeedForwardNetwork(input_dim=d_model, hidden_dim=hidden_dim)
        self.addnorm2 = AddNorm(dropout=addnorm_dropout)
    
    def forward(self, X, valid_lens):
        Y = self.attention(X, X, X, valid_lens)
        X = self.addnorm1(X, Y)
        Y = self.ffn(X)
        return self.addnorm2(X, Y)

# 解码器Transformer块
class DecoderTransformerBlock(nn.Module):
    def __init__(self, i, d_model=512, n_heads=8, hidden_dim=2048, attention_dropout=0.1, addnorm_dropout=0.1):
        super(DecoderTransformerBlock, self).__init__()
        self.i = i
        self.attention1 = MultiheadAttention(d_model=d_model, n_heads=n_heads, attention_dropout=0.1)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = MultiheadAttention(d_model=d_model, n_heads=n_heads, attention_dropout=0.1)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = FeedForwardNetwork(input_dim=d_model, hidden_dim=hidden_dim)
        self.addnorm3 = AddNorm(dropout=addnorm_dropout)

    def forward(self, x, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = x
        else:
            key_values = torch.cat((state[2][self.i], x), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, length, _ = x.shape
            dec_valid_lens = torch.arange(1, length + 1, device=x.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        y = attention1(x, key_values, key_values, dec_valid_lens)
        x = self.addnorm1(x, y)
        y = attention2(x, encoder_output, encoder_output, enc_valid_lens)
        x = self.addnorm2(x, y)
        y = self.ffn(x)
        return self.addnorm3(x, y), state

class Encoder(nn.Module):
    def __init__(self, d_model, vocab_size, max_tokens, num_layers, n_heads, hidden_dim=2048):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(d_model=self.d_model, dropout=0.1, max_tokens=max_tokens)
        self.layers = nn.Sequencial()
        for _ in range(num_layers):
            self.layers.add(EncoderTransformerBlock(d_model=self.d_model, n_heads=n_heads, hidden_dim=hidden_dim, attention_dropout=0.1, addnorm_dropout=0.1))
    
    def forward(self, x, valid_lens):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        self.attention_weight = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x, valid_lens)
            self.attention_weight[i] = layers.attention.attention.attention_weights
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, vocab_size, max_tokens, num_layers, n_heads, hidden_dim=2048):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(d_model=self.d_model, dropout=0.1, max_tokens=max_tokens)
        self.layers = nn.Sequencial()
        for _ in range(self.num_layers):
            self.layers.add(EncoderTransformerBlock(d_model=self.d_model, n_heads=n_heads, hidden_dim=hidden_dim, attention_dropout=0.1, addnorm_dropout=0.1))
        self.dense = nn.Linear(self.d_model, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    
    def forward(self, x, state):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        self.attention_weights = [[None] * len(self.layers) for _ in range (2)]
        for i, layer in enumerate(self.layers):
            x = layer(x, state)
            self.attention_weight[0][i] = layers.attention1.attention.attention_weights
            self.attention_weight[1][i] = layers.attention2.attention.attention_weights
        return self.dense(x), state
    

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)


class Transformer:
    def __init__(self, *args):
        super(Transformer, self).__init__()
        self.device = "cuda:0"
        self.model_set()
        self.training_set()
        
    
    def model_set(self, vocab_size, d_model=512, n_layers=[6,6], n_heads=8, num_hiddens=2048, dropout=0.1):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.model = EncoderDecoder(self.encoder, self.decoder)
        # 随机初始化参数
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        
    def training_set(self, epoch=10, lr=1e-3, loss_compute=None):
        pass

    def train(self, data_iter, tgt_vocab):
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(data_iter):
            out = self.model(batch.src, batch.trg)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % 100 == 1:
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
                start = time.time()
                tokens = 0
        return total_loss / total_tokens




if __name__ == '__main__':
    X = torch.ones((2,10,512))
    print("Position _encoding")
    positional_encoding = PositionalEncoding(512)
    positional_encoding.eval()
    X = positional_encoding(X)
    pos = positional_encoding.positional_encoding
    print(X.shape, pos.shape)
    attention = MultiheadAttention(512, 2)
    attention.eval()
    valid_lens = torch.tensor([3, 2])
    X = attention(X, X, X, valid_lens)
    print(X.shape)
    encoderblock = EncoderBlock()
    encoderblock.eval()
    X = encoderblock(X)
    print(X.shape)
