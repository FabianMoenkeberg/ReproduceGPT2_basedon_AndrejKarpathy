import torch
from torch.nn import functional as F

from model.model import GPT, GPTConfig
from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
sd_hf = model_hf.state_dict()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'

print("Using device:", device)

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained("gpt2")  # Load the GPT-2 model
model.eval()
model.to(device)

# print the generated text
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")  # Encode the input text
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # Repeat for num_return_sequences
x = tokens.to(device)

# generate!  right now x is (B, T) where B = 5, T = 8
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length: # max_length=30
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)[0] # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities -> to create different outputs
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
import tiktoken
enc = tiktoken.get_encoding('gpt2')
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

## Train on Shakespeare
# tiny shakespeare dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r') as f:
    text = f.read()
data = text[:1000] # first 1,000 characters
print(data[:100])

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(data)
print(tokens[:24])

# Create a tensor from the tokens. Split into 4 sequences of 6 tokens each
# For the labels we shift the tokens by one position. Thus, we start from the second token.
import torch
buf = torch.tensor(tokens[:24 + 1])
x = buf[:-1].view(4, 6)
y = buf[1:].view(4, 6)
print(x)
print(y)

# print(sd_hf["lm_head.weight"].shape)
# print(sd_hf["transformer.wte.weight"].shape)

# (sd_hf["lm_head.weight"] == sd_hf["transformer.wte.weight"]).all()

# print(sd_hf["lm_head.weight"].data_ptr())
# print(sd_hf["transformer.wte.weight"].data_ptr())

# # standard deviation grows inside the residual stream
# x = torch.zeros(768)
# n = 100 # e.g. 100 layers
# for i in range(n):
#     x += n**-0.5 * torch.randn(768)

# print(x.std())