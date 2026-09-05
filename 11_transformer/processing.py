import re
import unicodedata
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset


class Tokenizer:
    def __init__(self, lang):
        if lang not in ("en", "de"):
            raise ValueError("lang 必须为 en 或 de")
        self.lowercase = lang == "en"
        # Unicode 字母包括 é 等；保留缩写、数字和独立标点。
        number = r"\d+(?:[.,]\d+)?" if lang == "de" else r"\d+(?:\.\d+)?"
        self.pattern = re.compile(
            r"[^\W\d_]+(?:['’][^\W\d_]+)?|" + number + r"|\.\.\.|--|[^\w\s]|_"
        )

    def _tokenize(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFC", text)
        if self.lowercase:
            text = text.lower()
        return self.pattern.findall(text)

    def __call__(self, text):
        if isinstance(text, str):
            return self._tokenize(text)
        return [self._tokenize(t) for t in text]


class Vocab:
    """给文本建立字典."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        tokens = [] if tokens is None else tokens
        reserved_tokens = list(reserved_tokens) if reserved_tokens is not None else ["<eos>", "<bos>", "<padding>"]
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(
            sorted(
                set(
                    ["<unk>"]
                    + reserved_tokens
                    + [token for token, freq in self.token_freqs if freq >= min_freq]
                )
            )
        )
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        # 修改：按类型区分标量与序列，也支持 Tensor 和嵌套 batch。
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        if isinstance(indices, (list, tuple)):
            return [self.to_tokens(index) for index in indices]
        return self.idx_to_token[int(indices)]
    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

def build_array(tokenized_data, vocab, num_steps):
    """将分词文本转换为 PyTorch Tensor"""
    if num_steps < 2:
        raise ValueError("num_steps 至少为 2，为 BOS 和 EOS 留出位置")
    for token in ("<padding>", "<bos>", "<eos>"):
        if token not in vocab.token_to_idx:
            raise ValueError(f"词表缺少特殊标记 {token}")
    pad_idx = vocab['<padding>']
    bos_idx = vocab['<bos>']
    eos_idx = vocab['<eos>']
    
    # 修改：先截断正文，保证每条样本保留 EOS。
    lines = [[bos_idx] + vocab[tokens[:num_steps - 2]] + [eos_idx] for tokens in tokenized_data]
    array = [truncate_pad(line, num_steps, pad_idx) for line in lines]
    valid_len = [min(len(line), num_steps) for line in lines]
    
    return torch.tensor(array, dtype=torch.long).reshape(-1, num_steps), torch.tensor(valid_len, dtype=torch.long)


def load_multi30k(batch_size=32, num_steps=32, min_freq=2,
                 reserved_tokens=None, dataset=None, src_vocab=None,
                 tgt_vocab=None, seed=42):
    """英语 -> 德语。返回 (loaders, src_vocab, tgt_vocab)。

    loaders 包含 train/validation/test；每个 batch 为
    (src, src_valid_len, tgt, tgt_valid_len)，全部为 int64。
    src/tgt: (B, num_steps)，含 BOS/EOS 和右侧 padding。
    valid_len: (B,)，含 BOS/EOS、不含 padding。
    可以传入已经 load_dataset 得到的 dataset，避免重复加载。
    """
    if dataset is None:
        dataset = load_dataset("bentrevett/multi30k")
    if (src_vocab is None) != (tgt_vocab is None):
        raise ValueError("必须同时提供源词表和目标词表")
    en_tokenizer, de_tokenizer = Tokenizer("en"), Tokenizer("de")
    tokenized = {}
    for split in ("train", "validation", "test"):
        tokenized[split] = (
            en_tokenizer(dataset[split]["en"]),
            de_tokenizer(dataset[split]["de"]),
        )
    # 仅训练集参与建词表，其余 split 复用同一套 ID。
    if src_vocab is None:
        src_vocab = Vocab(tokenized["train"][0], min_freq, reserved_tokens)
        tgt_vocab = Vocab(tokenized["train"][1], min_freq, reserved_tokens)
    loaders = {}
    for split, (src_tokens, tgt_tokens) in tokenized.items():
        src, src_len = build_array(src_tokens, src_vocab, num_steps)
        tgt, tgt_len = build_array(tgt_tokens, tgt_vocab, num_steps)
        loaders[split] = DataLoader(
            TensorDataset(src, src_len, tgt, tgt_len),
            batch_size=batch_size, shuffle=(split == "train"),
            generator=torch.Generator().manual_seed(seed),
        )
    return loaders, src_vocab, tgt_vocab




# if __name__ == '__main__':
#     loaders, src_vocab, tgt_vocab = load_multi30k(min_freq=5)
#     print("词表大小:", len(src_vocab), len(tgt_vocab))
#     for tensor in next(iter(loaders["train"])):
#         print(tensor.shape, tensor.dtype)
