from transformer import Transformer
from data_load import dataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleLossCompute:
    "损失计算"
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        # print("x ", x.shape)
        # print("y ", y.shape)
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # print(loss.data , norm)
        return loss.data * norm


class NoamOpt:
    "warmup优化器"
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "更新参数与学习率"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def greedy_decode(model, src, src_mask, max_len=200, start_symbol=1):
    enc_output = model.encode(src, src_mask)
    dec = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(enc_output, src_mask, Variable(dec), Variable(subsequent_mask(dec.size(1)).type_as(src.data)))
        word_pro = model.generator(out[:, -1])
        _, next_word = torch.max(word_pro, dim = 1)
        next_word = next_word.data[0]
        dec = torch.cat([dec, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return dec    


def run_epoch(dataloader, model, loss_computer):
    t_start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, (enc_inputs, dec_inputs, dec_outputs, src_masks, tgt_masks, num_tokens) in enumerate(dataloader):
        
        # print(" enc_inputs ", enc_inputs.shape," dec_inputs ", dec_inputs.shape," dec_outputs ", dec_outputs.shape," src_masks ", src_masks.shape," tgt_masks ", tgt_masks.shape, num_tokens)
        enc_inputs, dec_inputs, dec_outputs, src_masks, tgt_masks, num_tokens = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device), src_masks.to(device), tgt_masks.to(device), num_tokens.to(device)
        output = model(enc_inputs, dec_inputs, src_masks, tgt_masks)
        
        loss = loss_computer(output, dec_outputs, num_tokens)
        
        total_loss += loss
        total_tokens += num_tokens
        tokens += num_tokens
        
        if i % 100 == 1:
            elapsed = time.time() - t_start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f"  %(i, loss / num_tokens, tokens / elapsed))
            t_start = time.time()
            tokens = 0
    return total_loss / total_tokens


def train(args, data, model, criterion, model_opt):
    """训练函数"""
    t_s = time.time()
    best_loss = 9999999
    for epoch in range(args.epochs):
        print("*******Epoch {} Start*******".format(epoch+1))
        model.train()
        train_loss = run_epoch(data.train_dataloader, model, SimpleLossCompute(model.generator, criterion, model_opt))
        print(">> Epoch:{}: train_loss: {}".format(epoch+1, valid_loss))
        model.eval()
        valid_loss = run_epoch(data.valid_dataloader, model, SimpleLossCompute(model.generator, criterion, None))
        print(">> Epoch:{}: valid_loss: {}".format(epoch+1, valid_loss))
        if valid_loss < best_loss:
            # 保存模型
            print("-- Saving Model in epoch:{}.".format(epoch+1))
            torch.save(model, args.model_dir+'/best_model.pt')
            print("-- Model Saved.")
        print("*******Epoch {} Finished, Spend time {} min*******".format(epoch+1, (time.time-t_s)/60))


def test(args, data, model, criterion, model_opt):
    model=torch.load(args.model_dir+'/best_model.pt')
    t_s = time.time()
    



    print(">> Epoch:{}: test_loss: {}".format(epoch+1, test_loss))
    print("Total time spended: {}".format(time.time() - t_s))


class Args():
    def __init__(self):
        super(Args, self).__init__()
        self.tag = "transformer"
        # data
        self.data_dir = "./nmt/en-zh/News-Commentary-small/data"
        self.src = "en"
        self.tgt = "zh"
        self.max_tokens = 2048
        self.batch_size = 64
        self.num_workers = 8
        # model
        self.model_dir = "./nmt/en-zh/News-Commentary-small/model"+"/self.tag"
        self.num_layers = [6,6]
        self.d_model = 512
        self.hidden_dim = 2048
        self.n_heads = 8
        self.dropout = [0.1, 0.1, 0.1, 0.1]
        # train
        self.epochs = 5
        self.warmup = 4000
        self.lr = 1e-3


def main():
    task = "train" # train / test

    args = Args()
    data = dataLoader(args.data_dir)
    model = Transformer(src_vocab=len(data.get_dict()[0]), tgt_vocab=len(data.get_dict()[1]), 
                        max_tokens=args.max_tokens, num_layers=args.num_layers, 
                        d_model=args.d_model, hidden_dim=args.hidden_dim, 
                        n_heads=args.n_heads, dropout=args.dropout).model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, momentum=0.98)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    model_opt = NoamOpt(args.d_model, 1, args.warmup, optimizer)

    print(model)
    if task == "train":
        train(args, data, model, criterion, model_opt)
    elif task == "test":
        test(args, data, model, criterion, model_opt)
    else:
        print("Error: Wrong task.")


if __name__ == '__main__':
    main()
