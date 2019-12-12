import time
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import torch


def sample_image(model, encoder, output_image_dir, n_row, batches_done, dataloader, device):
    """Saves a grid of generated imagenet pictures with captions"""
    target_dir = os.path.join(output_image_dir, "samples/")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    captions = []
    gen_imgs = []
    # get sample captions
    done = False
    while not done:
        for (_, labels_batch, captions_batch) in dataloader:
            captions += captions_batch
            conditional_embeddings = encoder(labels_batch.to(device), captions)
            imgs = model.sample(conditional_embeddings).cpu()
            gen_imgs.append(imgs)

            if len(captions) > n_row ** 2:
                done = True
                break

    gen_imgs = torch.cat(gen_imgs).numpy()
    gen_imgs = np.clip(gen_imgs, 0, 1)

    fig = plt.figure(figsize=((8, 8)))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row), axes_pad=0.2)

    for i in range(n_row ** 2):
        grid[i].imshow(gen_imgs[i].transpose([1, 2, 0]))
        grid[i].set_title(captions[i])
        grid[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=True)

    save_file = os.path.join(target_dir, "{:013d}.png".format(batches_done))
    plt.savefig(save_file)
    print("saved  {}".format(save_file))
    plt.close()


def sample_images_full(model, encoder, output_image_dir, dataloader, device):
    """Saves a grid of generated imagenet pictures with captions"""
    target_dir = os.path.join(output_image_dir, "full_samples/")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    start_time = time.time()
    gen_imgs = []
    for (_, labels_batch, captions_batch) in dataloader:
        conditional_embeddings = encoder(labels_batch.to(device), captions_batch)
        imgs = model.sample(conditional_embeddings).cpu().numpy()
        imgs = np.clip(imgs, 0, 1)
        imgs = np.split(imgs, imgs.shape[0])
        gen_imgs += imgs
    elapsed = time.time() - start_time
    print(elapsed)

    for i, img in enumerate(gen_imgs):
        img = np.squeeze(img)
        img = np.transpose(img, (1, 2, 0))
        save_file = os.path.join(target_dir, "{:013d}.png".format(i))
        matplotlib.image.imsave(save_file, img)
        print("saved  {}".format(save_file))


def save_imgs(output_image_dir, dataloader):
    """Saves a grid of generated imagenet pictures with captions"""
    target_dir = os.path.join(output_image_dir, "imgs/")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    i = 0
    for (imgs, _, _) in dataloader:
        imgs = imgs.cpu().numpy()
        imgs = np.clip(imgs, 0, 1)
        imgs = np.split(imgs, imgs.shape[0])
        for img in imgs:
            img = np.squeeze(img)
            img = np.transpose(img, (1, 2, 0))
            save_file = os.path.join(target_dir, "{:013d}.png".format(i))
            matplotlib.image.imsave(save_file, img)
            print("saved  {}".format(save_file))
            i += 1


def load_model(file_path, generative_model):
    dict = torch.load(file_path)
    generative_model.load_state_dict(dict)