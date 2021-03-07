"""
    LPCNet pytorch
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import argparse
import os
import time


from data_utils import FeaturePCMLoader
from hparams import create_hparams
from lpcnet_bunched import LPCNetModelBunch
from plotting_utils import LPCNetLogger, stream
from dump_lpcnet import convert_recurrent_kernel, re_convert_recurrent_kernel
from stft import MultiResolutionSTFTLoss
from ulaw import ulaw2lin

def prepare_logger(log_dir):
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logger = LPCNetLogger(log_dir)
    return logger

def prepare_dataloaders(hparams):
    train_set = FeaturePCMLoader(hparams.features, hparams.pcms, hparams.frame_size, hparams.nb_used_features,
                                 hparams.bfcc_band, hparams.nb_features, hparams.pitch_idx, hparams.n_samples_per_step, hparams.checkpoint_path)
    train_loader = DataLoader(train_set,
                              num_workers=4, shuffle=True,
                              batch_size=hparams.batch_size, pin_memory=True,
                              drop_last=False)

    validation_loader = None
    test_loader = None

    return train_loader, validation_loader, test_loader


def save_checkpoint(model, optimizer, learning_rate, epoc, checkpoint_path):
    print("\nSaving model and optimizer state at epoc {} to {}".format(
        epoc, checkpoint_path))
    torch.save(
        {'epoc':epoc,
         'state_dict': model.state_dict(),
         'optimizer':optimizer.state_dict(),
         'learning_rate':learning_rate},
        checkpoint_path
    )

def load_checkpoint(checkpoint_file, model, optimizer):
    checkpoint_dict = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    epoc = checkpoint_dict['epoc']
    learning_rate = checkpoint_dict["learning_rate"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print("Loaded checkpoint '{}' from epoc {}".format(
        checkpoint_file, epoc))
    return model, optimizer ,epoc


def validate(model, val_loader, criterion, iteration ,logger):
    model.eval()

    val_loss = 0
    for i, (x, y) in enumerate(val_loader):
        if None == y:
            break
        y_pred = model(x)
        y_pred = y_pred.permute(0, 2, 1).cuda()
        y = y.squeeze(2).cuda()
        loss = criterion(y_pred, y)
        val_loss += loss.item()
    val_loss = val_loss / len(val_loader)
    model.train()
    logger.log_validation(val_loss, iteration)

def sparse_gru_a(model, final_density, iteration, t_start, t_end):
    layer = model.gru_a
    # (3*H, H)
    w = layer.weight_hh_l0.cpu().data.numpy()
    p = w  # w[1]
    nb = p.shape[0] // p.shape[1]
    N = p.shape[1]
    # print("nb = ", nb, ", N = ", N);
    # print(p.shape)
    # print ("density = ", density)
    # p = np.transpose(p, (1, 0))
    p = convert_recurrent_kernel(p)
    for k in range(nb):
        density = final_density[k]
        if iteration < t_end:
            r = 1 - (iteration - t_start) / (t_end - t_start)
            density = 1 - (1 - final_density[k]) * (1 - r * r * r)
        A = p[:, k * N:(k + 1) * N]
        A = A - np.diag(np.diag(A))
        A = np.transpose(A, (1, 0))
        L = np.reshape(A, (N, N // 16, 16))
        S = np.sum(L * L, axis=-1)
        SS = np.sort(np.reshape(S, (-1,)))
        thresh = SS[round(N * N // 16 * (1 - density))]
        mask = (S >= thresh).astype('float32')
        mask = np.repeat(mask, 16, axis=1)
        mask = np.minimum(1, mask + np.diag(np.ones((N,))))
        mask = np.transpose(mask, (1, 0))
        p[:, k * N:(k + 1) * N] = p[:, k * N:(k + 1) * N] * mask
        # print(thresh, np.mean(mask))
    w = re_convert_recurrent_kernel(p)
    model.gru_a.weight_hh_l0.data = torch.from_numpy(w).cuda()


def train(args, hparams):
    model = LPCNetModelBunch(hparams).cuda()
    model.train()
    print("Model is \n", model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    print("r is ", hparams.n_samples_per_step)
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate, amsgrad=True)

    epoc_offset = 0
    if args.checkpoint is not None:
        model, optimizer, epoc_offset = load_checkpoint(args.checkpoint, model, optimizer)
        print("load checkpoint from ", args.checkpoint)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=(1 - hparams.lr_decay))
    criteon   = nn.CrossEntropyLoss().cuda()
    aux_criteon = MultiResolutionSTFTLoss().cuda()

    #logger
    logger = prepare_logger(hparams.log_dir)
    train_loader, val_loader, test_loader = prepare_dataloaders(hparams)

    iteration = 0
    for epoc in range(epoc_offset + 1, hparams.epochs):
        tot_loss = 0.0
        for i, (in_data, new_features, periods, target) in enumerate(train_loader, 1):
            in_data, new_features, periods, target = in_data.cuda(), new_features.cuda(), periods.cuda(), target.cuda()
            start = time.perf_counter()
            target_pred = model(in_data, new_features, periods, target)
            target_pred = target_pred.permute(0, 2, 1)
            target = target.squeeze(2)
            loss = criteon(target_pred, target)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()
            tot_loss = tot_loss + loss.item()
            avg_loss = tot_loss / i
            duration = time.perf_counter() - start
            if iteration % 10 == 0:
                logger.log_training("\nTrain", loss.item(), avg_loss, optimizer.param_groups[0]["lr"], duration, iteration)
                if iteration % 10 == 0:
                    logger.log_training("Train", loss.item(), avg_loss, optimizer.param_groups[0]["lr"], duration,
                                        iteration)
                    message = f'epoc: {epoc}/{hparams.epochs} | ({i}/{len(train_loader)}) | avg_loss: {avg_loss:#.4}'
                    stream(message)
            iteration += 1

            
            t_start = 2000
            t_end = 40000
            interval = 400
            density = (0.05, 0.05, 0.2)
            if iteration < t_start or ((iteration - t_start) % interval != 0 and iteration < t_end):
                # print("don't constrain")
                continue
            elif hparams.spartify:
                sparse_gru_a(model, density, iteration, t_start, t_end)

        os.makedirs(hparams.checkpoint_path, exist_ok=True)
        checkpoint_path = os.path.join(hparams.checkpoint_path, "pytorch_lpcnet20_384_10_G16_{:02d}.h5".format(epoc))
        save_checkpoint(model, optimizer, optimizer.param_groups[0]["lr"], epoc, checkpoint_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='checkpoint')
    args = parser.parse_args()
    hparams = create_hparams()

    print("training model")
    train(args, hparams)


if __name__=='__main__':
    main()
