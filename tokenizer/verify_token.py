
from pretraining.dataset import TextDataset

ds = TextDataset(
    root_path="data/raw",
    tokenizer_path="tokenizer/tokenizer_8k.json",
    block_size=128
)

print("Samples:", len(ds))

