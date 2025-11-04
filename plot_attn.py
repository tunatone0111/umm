import sys

sys.path.append("src/models/bagel")

import pickle
from PIL import Image
from src.models.bagel.data.transforms import ImageTransform
from src.models.bagel.modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer

import matplotlib.pyplot as plt


dname = "datacomp_small"
timestamp = "20251104_164004"

with open(f"{dname}_attention_weights_{timestamp}.pkl", "rb") as f:
    attention_weights = pickle.load(f)

original = Image.open(f"{dname}_original_{timestamp}.png")
edited = Image.open(f"{dname}_edited_{timestamp}.png")

with open(f"{dname}_prompt_{timestamp}.txt", "r") as f:
    prompt = f.read()

x = ImageTransform(1024, 512, 16)(original)
tokenizer = Qwen2Tokenizer.from_pretrained("data/models/BAGEL-7B-MoT")
tokens = tokenizer.tokenize(prompt)

ratio = x.shape[1] / x.shape[2]
scale = 3
ncols = 28
nrows = len(tokens) + 2
timestep = 30

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * scale, nrows * ratio * scale))

for k, ax_rows in enumerate(axes):
    for layer, ax in enumerate(ax_rows):
        ax.imshow(
            attention_weights[f"layer_{layer}_timestep_{timestep}"][0][:, 1:-1, k]
            .mean(axis=0)
            .reshape(
                x.shape[1] // 16,
                x.shape[2] // 16,
            ),
        )
        if k == 0:
            ax.set_title(f"layer {layer}")
        if layer == 0:
            ax.set_ylabel(["<bos>", *tokens, "<eos>"][k].replace("Ġ", ""))
        ax.set_xticks([])
        ax.set_yticks([])


fig.savefig(f"{dname}_attention_weights_{timestamp}_t{timestep}.pdf")
