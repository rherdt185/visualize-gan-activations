# Visualize and Paint GAN Activations

Visualize and Paint GAN Activations, to appear in MLSP 2024 [[arxiv](https://arxiv.org/pdf/2405.15636)]

`example.ipynb` shows an example on how to visualize an activation vector (a pixel out of a hidden layer of a GAN), and shows an example for a non-tileable and tileable feature.

`evaluation.py` generates random images, extracts a random activation vector from each of them, and sorts them by tileability (by how similar the visualizations generated with a block size of 2 are compared to setting all spatial activations to the activation vector are to a pretrained feature extractor). Finally visalizations for the 8 most and least tileable features are saved.



## Requirements

Depends on the StyleGAN2 repo, since we visualized features of those models. The path to the stylegan2 repo ([link](https://github.com/NVlabs/stylegan2-ada-pytorch)) needs to be set in `config.py`.
Further example.ipynb needs the afhqwild.pkl checkpoint present inside this repo (see load_gan inside `core.py`), which can also be downloaded from the stylegan2-ada-pytorch repo in the link above.
