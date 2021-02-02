# SIPEC
# MARKUS MARKS
# UNSUPERVISED ANALYSIS
from sklearn import decomposition
from sklearn.manifold import TSNE
from PIL import Image

import numpy as np
from tqdm import tqdm


results_path = "/media/nexus/storage4/swissknife_results/full_inference/mouse_test/"
results = np.load(results_path + "inference_results.npy", allow_pickle=True)

flattened_masks = []
sums = []

for el in tqdm(results[:30000]):
    for idx, _id in enumerate(el["ids"]):
        img = el["masked_masks"][idx]
        flattened_masks.append(img)
        sums.append(img.sum())

flattened_masks_flat = flattened_masks.reshape(
    (flattened_masks.shape[0], flattened_masks.shape[1] * flattened_masks.shape[1])
)

pca = decomposition.PCA(n_components=0.95)
pca.fit(flattened_masks_flat)
X = np.array(pca.transform(flattened_masks_flat))

tsne = TSNE(
    n_components=2, learning_rate=150, perplexity=20, angle=0.2, verbose=2
).fit_transform(X[2:])

tx, ty = tsne[:, 0], tsne[:, 1]
tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

width = 4000
height = 3000
max_dim = 100

# visualize

full_image = Image.new("RGBA", (width, height))
for img, x, y in zip(flattened_masks, tx, ty):
    img[img < 0.5] = 0  # Black
    img[img >= 0.5] = 255  # White
    tile = Image.fromarray(img)
    rs = max(1, tile.width / max_dim, tile.height / max_dim)
    tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
    full_image.paste(
        tile,
        (int((width - max_dim) * x), int((height - max_dim) * y)),
        mask=tile.convert("RGBA"),
    )
