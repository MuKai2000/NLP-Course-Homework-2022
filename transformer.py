import torch
import torch.nn as nn
import numpy as np 
import math
import matplotlib.pyplot as plt


def get_mask(size):
    "生成Mask"
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')    # 下三角矩阵
    return torch.from_numpy(mask) == 0


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, dropout=0.1, max_tokens=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)  # dropout

        self.positional_encoding = torch.zeros((1, max_tokens, d_model))    # 初始化位置编码矩阵
        i = torch.arange(max_tokens, dtype=torch.float32).reshape(-1, 1)
        x = i / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        self.positional_encoding[:, :, 0::2] = torch.sin(x)
        self.positional_encoding[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.shape[1], :].to(x.device)    # 添加位置编码
        return self.dropout(x)


class DotProductAttention(nn.Module):
    """计算点积注意力矩阵"""

    def __init__(self, dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
  
    def forward(query, key, value, mask=None):
        dim = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        attention_weight = nn.functional.softmax(scores, dim=-1)
        attention_weight = self.dropout(attention_weight)
        return torch.matmul(attention_weight, value), attention_weight


class MultiheadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.n_heads = n_heads   
        assert d_model % self.n_heads == 0
        # 线性变换层
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.attention = DotProductAttention(dropout)
        self.fc = nn.Linear(d_model, d_model)
        
    def transpose(self, x, n_heads):
        # X : Batch * Length * size
        x = x.reshape(x.shape[0], x.shape[1], n_heads, -1) # Batch * Length * N_head * size//N_head
        x = x.permute(0, 2, 1, 3) # Batch * N_head * Length * size//N_head
        return x.reshape(-1, x.shape[2], x.shape[3]) # (Batch * N_head) * Length * size//N_head

    def retranspose(self, x, n_heads):
        x = x.reshape(-1, n_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1) # Batch * Length * size

    def forward(self, query, key, value, mask=None):
        query = self.transpose(self.q(query), self.n_heads) # (Batch * N_head) * Length * num_hidden
        key= self.transpose(self.k(key), self.n_heads)
        value = self.transpose(self.v(value), self.n_heads)
        
        if mask is not None:
            # 处理匹配多头
            mask = mask.unsqueeze(1)
        
        x, attetion_weight = self.attention(query, key, value, mask) # (Batch * N_head) * Length * D_model
        output_concat = self.retranspose(output, self.n_heads)
        return self.fc(output_concat), attetion_weight


class FeedForwardNetwork(nn.Module):
    """前馈神经网络"""

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class AddNorm(nn.Module):
    """残差连接&层归一化"""

    def __init__(self, dropout=0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, y):
        x = self.dropout(x + y)
        layer_norm = nn.LayerNorm(x.shape[1:])
        return layer_norm(x)


class EncoderBlock(nn.Module):
    """Encoder块"""

    def __init__(self, d_model=512, n_heads=8, hidden_dim=2048, attention_dropout=0.1, ffn_dropout=0.1, addnorm_dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=attention_dropout)    # Attention层
        self.addnorm1 = AddNorm(dropout=addnorm_dropout)    # 残差链接与归一化
        self.ffn = FeedForwardNetwork(input_dim=d_model, hidden_dim=hidden_dim, dropout=ffn_dropout) # 前馈神经网路
        self.addnorm2 = AddNorm(dropout=addnorm_dropout)    # 残差链接与归一化
    
    def forward(self, x, mask):
        y, _ = self.attention(x, x, x, mask)
        x = self.addnorm1(x, y)
        y = self.ffn(x)
        return self.addnorm2(x, y)


class DecoderBlock(nn.Module):
    """Decoder块"""

    def __init__(self, i, d_model=512, n_heads=8, hidden_dim=2048, attention_dropout=0.1, ffn_dropout=0.1, addnorm_dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.i = i  # 当前块序号
        self.attention1 = MultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=attention_dropout)   # 注意力层
        self.addnorm1 = AddNorm(dropout=addnorm_dropout)   # 残差连接与归一化
        self.attention2 = MultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=attention_dropout)   # 注意力层
        self.addnorm2 = AddNorm(dropout=addnorm_dropout)   # 残差连接与归一化
        self.ffn = FeedForwardNetwork(input_dim=d_model, hidden_dim=hidden_dim, dropout=ffn_dropout)    # 前馈层
        self.addnorm3 = AddNorm(dropout=addnorm_dropout)    # 残差连接与归一化

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """前向传播"""
        y, _ = attention1(x, x, x, tgt_mask)
        x = self.addnorm1(x, y)
        y, _ = attention2(x, enc_output, enc_output, src_mask)
        x = self.addnorm2(x, y)
        y = self.ffn(x)
        return self.addnorm3(x, y)


class Encoder(nn.Module):
    """编码器"""

    def __init__(self, d_model, vocab_size, max_tokens, num_layers=6, n_heads=8, hidden_dim=2048, dropout=[0.1,0.1,0.1,0.1]):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, self.d_model) # Embedding层
        self.pos_encoding = PositionalEncoding(d_model=self.d_model, dropout=dropout[0], max_tokens=max_tokens)    # 位置编码
        self.layers = nn.Sequential()   
        for _ in range(num_layers):
            self.layers.append(EncoderBlock(d_model=self.d_model, n_heads=n_heads, hidden_dim=hidden_dim, attention_dropout=dropout[1], ffn_dropout=dropout[2], addnorm_dropout=dropout[3])) # Encoder块
    
    def forward(self, x, mask):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))  # 通过Embedding与位置编码
        for layer in self.layers:
            x = layer(x, mask)  # num_layer层Encoder块
        return x


class Decoder(nn.Module):
    """解码器"""

    def __init__(self, d_model, vocab_size, max_tokens, num_layers=6, n_heads=8, hidden_dim=2048, dropout=[0.1,0.1,0.1,0.1]):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, self.d_model) # Embedding层
        self.pos_encoding = PositionalEncoding(d_model=self.d_model, dropout=dropout[0], max_tokens=max_tokens)    # 位置编码
        self.layers = nn.Sequential()
        for i in range(self.num_layers):
            self.layers.append(DecoderBlock(i=i, d_model=self.d_model, n_heads=n_heads, hidden_dim=hidden_dim, attention_dropout=dropout[1], ffn_dropout=dropout[2], addnorm_dropout=dropout[3])) # 解码器层

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        for i, layer in enumerate(self.layers):
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x


class Generator(nn.Module):
    """生成器：将解码器的输出映射至词表维度"""

    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.net = nn.Linear(d_model, vocab_size)   # 线性变换层
    
    def forward(self, x):
        return nn.functional.softmax(self.net(x), dim=-1)


class EncoderDecoder(nn.Module):
    """编码器解码器结构"""

    def __init__(self, encoder, decoder, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder  # 编码器
        self.decoder = decoder  # 编码器
        self.generator = generator  # 生成器 解码器输出映射至词表
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """前向传播""" 
        enc_output = self.encoder(src, src_mask) # encoder输出
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask) # decoder输出
        return dec_output


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, max_tokens, num_layers=[6,6], d_model=512, hidden_dim=2048, n_heads=8, dropout=[0.1,0.1,0.1,0.1]):
        super(Transformer, self).__init__()
        
        self.model = EncoderDecoder(encoder=Encoder(d_model=d_model, vocab_size=src_vocab, max_tokens=max_tokens, num_layers=num_layers[0], n_heads=n_heads, hidden_dim=hidden_dim, dropout=dropout), 
                                    decoder=Decoder(d_model=d_model, vocab_size=tgt_vocab, max_tokens=max_tokens, num_layers=num_layers[1], n_heads=n_heads, hidden_dim=hidden_dim, dropout=dropout), 
                                    generator=Generator(d_model=d_model, vocab_size=tgt_vocab))
        # 初始化参数
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

transformer = Transformer(20, 20, 10)
model = transformer.model
print(model)