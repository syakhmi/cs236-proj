import argparse
import os
from data import CIFAR10Dataset, Imagenet32DatasetDiscrete
from models.embedders import BERTEncoder, OneHotClassEmbedding, UnconditionalClassEmbedding
import torch
from utils.utils import sample_image, load_model, sample_images_full
from torch.optim import lr_scheduler
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np

from models import FlowPlusPlus
import util

os.makedirs("images", exist_ok=True)

def str2bool(s):
    return s.lower().startswith('t')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument('--lr_decay', type=float, default=5e-5,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_resnet", type=int, default=5, help="number of layers for the pixelcnn model")
parser.add_argument("--n_filters", type=int, default=96, help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--use_cuda", type=int, default=1, help="use cuda if available")
parser.add_argument("--output_dir", type=str, default="outputs/flowpp", help="directory to store the sampled outputs")
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--train_on_val", type=int, default=0, help="train on val set, useful for debugging")
parser.add_argument("--train", type=int, default=1, help="0 = eval, 1=train")
parser.add_argument("--model_checkpoint", type=str, default=None,
                    help="load model from checkpoint, model_checkpoint = path_to_your_pixel_cnn_model.pt")
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument("--dataset", type=str, default="cifar10", choices=["imagenet32", "cifar10"])
parser.add_argument("--conditioning", type=str, default="unconditional", choices=["unconditional", "one-hot", "bert"])

# FlowPP Params
parser.add_argument('--drop_prob', type=float, default=0.2, help='Dropout probability')
parser.add_argument('--num_blocks', default=10, type=int, help='Number of blocks in Flow++')
parser.add_argument('--num_components', default=32, type=int, help='Number of components in the mixture')
parser.add_argument('--num_dequant_blocks', default=2, type=int, help='Number of blocks in dequantization')
parser.add_argument('--use_attn', type=str2bool, default=True, help='Use attention in the coupling layers')
parser.add_argument('--max_grad_norm', type=float, default=1., help='Max gradient norm for clipping')
parser.add_argument('--warm_up', type=int, default=200, help='Number of batches for LR warmup')

# My Params
parser.add_argument('--train_size', type=int, default=5000, help='Number of training items')
parser.add_argument('--val_size', type=int, default=1000, help='Number of val items')
parser.add_argument('--val_start_idx', type=int, default=0, help='Starting index for validations')

global_step = 0


def train(model, embedder, optimizer, scheduler,
          train_loader, val_loader, opt, writer, device=None):
    print("TRAINING STARTS")
    global global_step
    for epoch in range(opt.n_epochs):
        print("[Epoch %d/%d]" % (epoch + 1, opt.n_epochs))
        model = model.train()
        loss_to_log = 0.0
        loss_fn = util.NLLLoss().to(device)
        with tqdm(total=len(train_loader.dataset)) as progress_bar:
            for i, (imgs, labels, captions) in enumerate(train_loader):
                start_batch = time.time()
                imgs = imgs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    if opt.conditioning == 'unconditional':
                        condition_embd = None
                    else:
                        condition_embd = embedder(labels, captions)

                optimizer.zero_grad()

                # outputs = model.forward(imgs, condition_embd)
                # loss = outputs['loss'].mean()
                # loss.backward()
                # optimizer.step()
                z, sldj = model.forward(imgs, condition_embd, reverse=False)
                loss = loss_fn(z, sldj) / np.prod(imgs.size()[1:])
                loss.backward()
                if opt.max_grad_norm > 0:
                    util.clip_grad_norm(optimizer, opt.max_grad_norm)
                optimizer.step()
                scheduler.step(global_step)

                batches_done = epoch * len(train_loader) + i
                writer.add_scalar('train/bpd', loss / np.log(2), batches_done)
                loss_to_log += loss.item()
                # if (i + 1) % opt.print_every == 0:
                #     loss_to_log = loss_to_log / (np.log(2) * opt.print_every)
                    #     print(
                    #         "[Epoch %d/%d] [Batch %d/%d] [bpd: %f] [Time/batch %.3f]"
                    #         % (epoch + 1, opt.n_epochs, i + 1, len(train_loader), loss_to_log, time.time() - start_batch)
                    #     )
                progress_bar.set_postfix(
                         bpd=(loss_to_log / np.log(2)),
                         lr=optimizer.param_groups[0]['lr'])
                progress_bar.update(imgs.size(0))
                global_step += imgs.size(0)

                loss_to_log = 0.0

                if (batches_done + 1) % opt.sample_interval == 0:
                    print("sampling_images")
                    model = model.eval()
                    sample_image(model, embedder, opt.output_dir, n_row=4,
                                 batches_done=batches_done,
                                 dataloader=val_loader, device=device)

        val_bpd = eval(model, embedder, val_loader, opt, writer, device=device)
        writer.add_scalar("val/bpd", val_bpd, (epoch + 1) * len(train_loader))

        torch.save(model.state_dict(),
                   os.path.join(opt.output_dir, 'models', 'epoch_{}.pt'.format(epoch)))


def eval(model, embedder, test_loader, opt, writer, device=None):
    print("EVALUATING ON VAL")
    model = model.eval()
    bpd = 0.0
    loss_fn = util.NLLLoss().to(device)
    for i, (imgs, labels, captions) in tqdm(enumerate(test_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            if opt.conditioning == 'unconditional':
                condition_embd = None
            else:
                condition_embd = embedder(labels, captions)

            # outputs = model.forward(imgs, condition_embd)
            # loss = outputs['loss'].mean()
            z, sldj = model.forward(imgs, condition_embd, reverse=False)
            loss = loss_fn(z, sldj) / np.prod(imgs.size()[1:])

            bpd += loss / np.log(2)
    bpd /= len(test_loader)
    print("VAL bpd : {}".format(bpd))
    return bpd


def main(args=None):
    if args:
        opt = parser.parse_args(args)
    else:
        opt = parser.parse_args()

    print(opt)

    print("loading dataset")
    if opt.dataset == "imagenet32":
        train_dataset = Imagenet32DatasetDiscrete(train=not opt.train_on_val, max_size=1 if opt.debug else opt.train_size)
        val_dataset = Imagenet32DatasetDiscrete(train=0, max_size=1 if opt.debug else opt.val_size, start_idx=opt.val_start_idx)
    else:
        assert opt.dataset == "cifar10"
        train_dataset = CIFAR10Dataset(train=not opt.train_on_val, max_size=1 if opt.debug else -1)
        val_dataset = CIFAR10Dataset(train=0, max_size=1 if opt.debug else -1)

    print("creating dataloaders")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
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

    # generative_model = ConditionalPixelCNNpp(embd_size=encoder.embed_size, img_shape=train_dataset.image_shape,
    #                                          nr_resnet=opt.n_resnet, nr_filters=opt.n_filters,
    #                                          nr_logistic_mix=3 if train_dataset.image_shape[0] == 1 else 10)

    generative_model = FlowPlusPlus(scales=[(0, 4), (2, 3)],
                                    # in_shape=(3, 32, 32),
                                    in_shape=train_dataset.image_shape,
                                    mid_channels=opt.n_filters,
                                    num_blocks=opt.num_blocks,
                                    num_dequant_blocks=opt.num_dequant_blocks,
                                    num_components=opt.num_components,
                                    use_attn=opt.use_attn,
                                    drop_prob=opt.drop_prob,
                                    condition_embd_size=encoder.embed_size)

    generative_model = generative_model.to(device)
    encoder = encoder.to(device)
    print("Models loaded on device")

    # Configure data loader

    print("dataloaders loaded")
    # Optimizers
    # optimizer = torch.optim.Adam(generative_model.parameters(), lr=opt.lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.lr_decay)
    param_groups = util.get_param_groups(generative_model, opt.lr_decay, norm_suffix='weight_g')
    optimizer = torch.optim.Adam(param_groups, lr=opt.lr)
    warm_up = opt.warm_up * opt.batch_size
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda s: min(1., s / warm_up))
    # create output directory

    os.makedirs(os.path.join(opt.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "tensorboard"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, "tensorboard"))

    global global_step
    global_step = 0

    # ----------
    #  Training
    # ----------
    if opt.train:
        train(model=generative_model, embedder=encoder, optimizer=optimizer, scheduler=scheduler,
              train_loader=train_dataloader, val_loader=val_dataloader, opt=opt, writer=writer, device=device)
    else:
        assert opt.model_checkpoint is not None, 'no model checkpoint specified'
        print("Loading model from state dict...")
        load_model(opt.model_checkpoint, generative_model)
        print("Model loaded.")
        sample_images_full(generative_model, encoder, opt.output_dir, dataloader=val_dataloader, device=device)
        eval(model=generative_model, embedder=encoder, test_loader=val_dataloader, opt=opt, writer=writer, device=device)

if __name__ == "__main__":
    main()
