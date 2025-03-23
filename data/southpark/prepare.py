import os
import requests
import pickle
import tiktoken
import numpy as np
import torch

enc = tiktoken.get_encoding("gpt2") #"o200k_base")

# Funtion to load the data from the ipnput.txt file into a string
def load_data():
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

text=load_data()
num_tokens_from_string(text, "o200k_base")

def encode(text:str):
    return enc.encode(text)

def decode(tokens:np.array):
    return enc.decode(tokens)

vocab_size = 200000

# encode with tiktoken gpt4 
data = torch.tensor(encode(text), dtype=torch.long) # Encoding of the entire text, sotring it in a torch tensor
print(f"length of dataset in tokens: {len(data):,}")
n = int(0.9*len(data)) # Number of characters to use for training
train_ids = data[:n]
val_ids = data[n:]
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
vocab_size = 200000

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint32)
val_ids = np.array(val_ids, dtype=np.uint32)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)