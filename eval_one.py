import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn import functional as F

import os
from datasets import NLMCXR, NLMCXR_ONE
from utils import data_to_device, data_concat, load, test_once
from models import CNN, MVCNN, TNN, Classifier, Generator, Clsgen, ClsGenInt


if __name__ == "__main__":
    path = os.environ.get("DemoPath")
    checkpoint_path_from = path + "\\checkpoints\\NLMCXR_Transformer_MaxView2_NumLabel114.pt"
    KW_SRC = ["txt"]
    LR = 5e-4
    WD = 1e-2
    NUM_EMBEDS = 256
    NUM_HEADS = 8
    FWD_DIM = 256
    VOCAB_SIZE = 1000
    POSIT_SIZE = 1000
    NUM_LABELS = 114
    NUM_CLASSES = 2
    NUM_LAYERS = 1
    DROPOUT = 0.1
    MILESTONES = [10]
    tnn = TNN(
        embed_dim=NUM_EMBEDS,
        num_heads=NUM_HEADS,
        fwd_dim=FWD_DIM,
        dropout=DROPOUT,
        num_layers=NUM_LAYERS,
        num_tokens=VOCAB_SIZE,
        num_posits=POSIT_SIZE,
    )
    model = Classifier(
        num_topics=NUM_LABELS,
        num_states=NUM_CLASSES,
        cnn=None,
        tnn=tnn,
        embed_dim=NUM_EMBEDS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    )
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WD,
    )
    model = nn.DataParallel(model).cuda()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)
    last_epoch, (best_metric, test_metric) = load(
        checkpoint_path_from, model, optimizer, scheduler
    )
    test_data_one = NLMCXR_ONE(
        directory=path + "\\NLMCXR\\",
        random_transform=True,
    )
    test_loader_once = data.DataLoader(
        test_data_one,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )
    outputs, targets = test_once(
        test_loader_once,
        model,
        device="cuda",
        kw_src=KW_SRC,
    )
    newoutputs = [float(outputs.cpu()[..., i, 1] > 0.5) for i in range(114)]
    newtargets = targets.cpu().squeeze(0)[0:114].tolist()
    print(newoutputs)
    print(newtargets)
