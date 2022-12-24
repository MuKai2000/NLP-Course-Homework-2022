from torch.utils.data import Dataset, DataLoader
import torch
from torch.autograd import Variable
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PairDataSet(Dataset):
    """数据集加载"""
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(PairDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
    
    def __len__(self):
        return len(self.enc_inputs)

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


def get_mask(size):
    "生成Mask下三角矩阵"
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')    # 下三角矩阵
    return torch.from_numpy(mask) == 0


def make_std_mask(tgt, pad=0):
    "生成Mask"
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(get_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


def pad_sequence(sequence, batch_first=False, max_len=None, padding_value=0):
    """对同一批次数据进行padding"""
    # print(type(sequence))
    if max_len is None:
        max_len = max([seq.size(0) for seq in sequence])
    out_tensors = []
    for seq in sequence:
        if seq.size(0) < max_len:
            seq = torch.cat([seq, torch.tensor([padding_value] * (max_len - seq.size(0)))], dim=0)
        else:
            seq = seq[:max_len]
        out_tensors.append(seq)
        # print(seq)
        # print(seq.size(0))
    # out_tensors = torch.LongTensor(out_tensors)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensor


def generate_batch(data_batch):
    """batch生成函数,进行padding"""
    # print(len(data_batch))
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for (enc_in, dec_in, dec_out) in data_batch:
        enc_inputs.append(torch.LongTensor(enc_in))
        dec_inputs.append(torch.LongTensor(dec_in))
        dec_outputs.append(torch.LongTensor(dec_out))

    enc_inputs  = pad_sequence(enc_inputs,batch_first=True, padding_value=0)
    dec_inputs  = pad_sequence(dec_inputs,batch_first=True, padding_value=0)
    dec_outputs  = pad_sequence(dec_outputs,batch_first=True, padding_value=0)
    
    # 生成Mask
    src_masks = (enc_inputs != 0).unsqueeze(-2)
    tgt_masks = make_std_mask(dec_inputs, 0)
    num_tokens = (dec_outputs != 0).data.sum()

    # print(enc_inputs[0], dec_inputs[0], dec_outputs[0], src_masks[0], tgt_masks[0], num_tokens)
    return enc_inputs, dec_inputs, dec_outputs, src_masks, tgt_masks, num_tokens


class dataLoader():
    """加载数据"""
    def __init__(self, data_dir, src='en', tgt='zh', batch_size=24, shuffle=False, num_workers=1):
        super(dataLoader, self).__init__()
        self.src = src
        self.tgt = tgt
        self.path = data_dir

        self.padding = "<PAD>"  # 0
        self.start = "<START>"  # 1
        self.end = "<END>"      # 2

        self.data = self.read_data(train_path=self.path + '/train',
                                   valid_path=self.path + '/valid', 
                                   test_path=self.path + '/test', 
                                   src_dict_path=self.path + '/data-bin/dict.' + self.src + '.txt', 
                                   tgt_dict_path=self.path + '/data-bin/dict.' + self.tgt + '.txt')
        
        self.data['train'] = self.make_data(self.data['train'], self.data['src_dict'], self.data['tgt_dict'])
        self.data['valid'] = self.make_data(self.data['valid'], self.data['src_dict'], self.data['tgt_dict'])
        self.data['test'] = self.make_data(self.data['test'], self.data['src_dict'], self.data['tgt_dict'])

        self.train_dataloader = DataLoader(PairDataSet(self.data['train']['enc_inputs'], 
                                                       self.data['train']['dec_inputs'], 
                                                       self.data['train']['dec_outputs']), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=generate_batch)
        self.valid_dataloader = DataLoader(PairDataSet(self.data['valid']['enc_inputs'], 
                                                       self.data['valid']['dec_inputs'], 
                                                       self.data['valid']['dec_outputs']), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=generate_batch)
        self.test_dataloader = DataLoader(PairDataSet(self.data['test']['enc_inputs'], 
                                                       self.data['test']['dec_inputs'], 
                                                       self.data['test']['dec_outputs']), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=generate_batch)

    def get_dict(self):
        return self.data["src_dict"], self.data["tgt_dict"], self.data["idx2word"]
    
    def get_dataloader(self):
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader

    def make_data(self, data, src_dict, tgt_dict):
        """将单词转换成字典索引"""
        enc_inputs, dec_inputs, dec_outputs = [], [], []
        for enc_in, dec_in, dec_out in data:
            enc_inputs.append([src_dict.get(word, 0) for word in enc_in])
            dec_inputs.append([tgt_dict.get(word, 0) for word in dec_in])
            dec_outputs.append([tgt_dict.get(word, 0) for word in dec_out])
        new_data = {'enc_inputs':enc_inputs, 'dec_inputs':dec_inputs, 'dec_outputs':dec_outputs}
        return new_data

    def read_pair(self, path, sorted=False):
        """读取语对数据"""
        src = [ line.strip('\n').split(' ') for line in open(path + "." + self.src).readlines()]
        tgt = [ line.strip('\n').split(' ') for line in open(path + "." + self.tgt).readlines()]
        data = []
        if sorted:
            for i in range(len(src)):
                data.append([src[i], [self.start] + tgt[i], tgt[i] + [self.end]])
            data.sort(key=lambda x:len(x[0]))
            """for line in data:
                print(len(line[0]), len(line[1]))"""
        print(" > ", path, len(data))
        return data

    def read_dict(self, path, tgt=False):
        """读取词典数据"""
        tag_dict = [self.padding]
        if tgt:
            tag_dict.append(self.start)
            tag_dict.append(self.end)
        dict = [ line.strip('\n').split(' ')[0] for line in open(path).readlines()]
        dict = tag_dict + dict
        print(" > ", path, len(dict))
        # 制作字典
        word_dict = {}
        idx2word = {}
        for i, word in enumerate(dict):
            word_dict[word] = i
            idx2word[i] = word
        return word_dict, idx2word
        
    def read_data(self, train_path, valid_path, test_path, src_dict_path, tgt_dict_path):
        """读取数据"""
        train = self.read_pair(train_path, True)
        valid = self.read_pair(valid_path, True)
        test = self.read_pair(test_path, True)

        src_dict, _ = self.read_dict(src_dict_path)
        tgt_dict, idx2word = self.read_dict(tgt_dict_path, tgt=True)
        
        return {"train":train, "valid":valid, "test":test, "src_dict":src_dict, "tgt_dict":tgt_dict, "idx2word":idx2word}

"""data = dataLoader("./nmt/en-zh/TED2013-small/data")

count = 0
for train_batch in data.train_dataloader:
    count += 1
    if count == 1:
        break"""