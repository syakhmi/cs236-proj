import argparse
import os
from data import CIFAR10Dataset, Imagenet32Dataset
from models.embedders import BERTEncoder, OneHotClassEmbedding, UnconditionalClassEmbedding
import torch
from models.cgan import CDCGAN_G, CDCGAN_D
from utils.utils import sample_image, load_model
from torch.optim import lr_scheduler
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument('--lr_decay', type=float, default=0.99,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_filters", type=int, default=128, help="num of filters in generative model")
parser.add_argument("--z_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--use_cuda", type=int, default=1, help="use cuda if available")
parser.add_argument("--output_dir", type=str, default="outputs/pixelcnn", help="directory to store the sampled outputs")
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--train_on_val", type=int, default=0, help="train on val set, useful for debugging")
parser.add_argument("--train", type=int, default=1, help="0 = eval, 1=train")
parser.add_argument("--model_checkpoint", type=str, default=None,
                    help="load model from checkpoint, model_checkpoint = path_to_your_pixel_cnn_model.pt")
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument("--dataset", type=str, default="cifar10", choices=["imagenet32", "cifar10"])
parser.add_argument("--conditioning", type=str, default="one-hot", choices=["unconditional", "one-hot", "bert"])


def train(model_G, model_D, embedder, optimizer_G, optimizer_D, scheduler_G, scheduler_D,
          train_loader, val_loader, adv_loss, opt):
    print("TRAINING STARTS")
    for epoch in range(opt.n_epochs):
        model_G = model_G.train()
        model_D = model_D.train()
        loss_G_to_log = 0.0
        loss_D_to_log = 0.0
        valid = torch.FloatTensor(opt.batch_size, 1).fill_(1.0).to(device)
        fake = torch.FloatTensor(opt.batch_size, 1).fill_(0.0).to(device)
        for i, (imgs, labels, captions) in enumerate(train_loader):
            start_batch = time.time()
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                condition_embd = embedder(labels, captions)
                condition_embd = condition_embd.unsqueeze(dim=2).unsqueeze(dim=3)
                condition_embd_expand = condition_embd.expand(-1, -1, 32, 32)

            # -----------------
            #  Train Generator
            # -----------------     
            optimizer_G.zero_grad()
            z = torch.randn(opt.batch_size, opt.z_dim, 1, 1).to(device)
            gen_imgs = model_G.forward(z, condition_embd)
            validity = model_D(gen_imgs, condition_embd_expand)
            loss_G = adv_loss(validity, valid)
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = model_D(imgs, condition_embd_expand)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = model_D(gen_imgs.detach(), condition_embd_expand)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            loss_D = (d_real_loss + d_fake_loss) / 2

            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------     
            optimizer_G.zero_grad()
            z = torch.randn(opt.batch_size, opt.z_dim, 1, 1).to(device)
            gen_imgs = model_G.forward(z, condition_embd)
            validity = model_D(gen_imgs, condition_embd_expand)
            loss_G = adv_loss(validity, valid)
            loss_G.backward()
            optimizer_G.step()

            batches_done = epoch * len(train_loader) + i
            writer.add_scalar('train/loss_G', loss_G.item(), batches_done)
            writer.add_scalar('train/loss_D', loss_D.item(), batches_done)
            loss_G_to_log += loss_G.item()
            loss_D_to_log += loss_D.item()
            if (i + 1) % opt.print_every == 0:
                loss_G_to_log = loss_G_to_log / opt.print_every
                loss_D_to_log = loss_D_to_log / opt.print_every
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [Loss G: %f] [Loss D: %f] [Time/batch %.3f]"
                    % (epoch + 1, opt.n_epochs, i + 1, len(train_loader), loss_G_to_log, loss_D_to_log, time.time() - start_batch)
                )
                loss_G_to_log = 0.0
                loss_D_to_log = 0.0
            if (batches_done + 1) % opt.sample_interval == 0:
                print("sampling_images")
                model_G = model_G.eval()
                sample_image(model_G, embedder, opt.output_dir, n_row=4,
                             batches_done=batches_done,
                             dataloader=val_loader, device=device)
                model_G = model_G.train()

        torch.save({'G':model_G.state_dict(), 'D':model_D.state_dict()},
                   os.path.join(opt.output_dir, 'models', 'epoch_{}.pt'.format(epoch)))

    scheduler_G.step()
    scheduler_D.step()

if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)

    print("loading dataset")
    if opt.dataset == "imagenet32":
        train_dataset = Imagenet32Dataset(train=not opt.train_on_val, max_size=1 if opt.debug else -1)
        val_dataset = Imagenet32Dataset(train=0, max_size=1 if opt.debug else -1)
    else:
        assert opt.dataset == "cifar10"
        train_dataset = CIFAR10Dataset(train=not opt.train_on_val, max_size=1 if opt.debug else -1)
        val_dataset = CIFAR10Dataset(train=0, max_size=1 if opt.debug else -1)

    print("creating dataloaders")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True
    )

    print("Len train : {}, val : {}".format(len(train_dataloader), len(val_dataloader)))

    device = torch.device("cuda") if (torch.cuda.is_available() and opt.use_cuda) else torch.device("cpu")
    print("Device is {}".format(device))

    print("Loading models on device...")

    # Initialize embedder
    if opt.conditioning == 'unconditional':
        encoder = UnconditionalClassEmbedding()
    elif opt.conditioning == "bert":
        encoder = BERTEncoder()
    else:
        assert opt.conditioning == "one-hot"
        encoder = OneHotClassEmbedding(train_dataset.n_classes)

    model_G = CDCGAN_G(z_dim=opt.z_dim, n_filters=opt.n_filters)
    model_D = CDCGAN_D(n_filters=opt.n_filters)
    model_G.weight_init(mean=0.0, std=0.02)
    model_D.weight_init(mean=0.0, std=0.02)

    model_G = model_G.to(device)
    model_D = model_D.to(device)
    encoder = encoder.to(device)
    # Loss functions
    adversarial_loss = torch.nn.MSELoss()
    adversarial_loss.cuda()
    print("Models loaded on device")

    # Configure data loader

    print("dataloaders loaded")
    # Optimizers
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=opt.lr)
    scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=1, gamma=opt.lr_decay)
    scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=1, gamma=opt.lr_decay)
    # create output directory

    os.makedirs(os.path.join(opt.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "tensorboard"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, "tensorboard"))

    # ----------
    #  Training
    # ----------
    if opt.train:
        train(model_G=model_G, model_D=model_D, embedder=encoder, optimizer_G=optimizer_G, optimizer_D=optimizer_D,
              scheduler_G=scheduler_G, scheduler_D=scheduler_D, train_loader=train_dataloader, val_loader=val_dataloader, adv_loss=adversarial_loss, opt=opt)
    else:
        assert opt.model_checkpoint is not None, 'no model checkpoint specified'
        print("Loading model from state dict...")
        load_model(opt.model_checkpoint, model_G)
        print("Model loaded.")
        evaluate(model_G=model_G, embedder=encoder, test_loader=val_dataloader)
