import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm
from datasets import NLMCXR, NLMCXR_ONE
from utils import data_to_device, data_concat, load, test_once
from models import CNN, MVCNN, TNN, Classifier, Generator, Clsgen, ClsGenInt
from losses import CELossTotalEval


def infer(data_loader, model, device="cpu", threshold=None):
    model.eval()
    outputs = []
    targets = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader, desc="Infer", leave=False)
        for source, target in prog_bar:
            source = data_to_device(source, device)
            target = data_to_device(target, device)

            if threshold is not None:
                '''output = model(image=source[0], history=source[3],caption=source[1], threshold=threshold)'''
                output = model(
                    image=source[0],
                    history=source[3],
                    threshold=threshold,
                )
            else:
                output = model(image=source[0])

            outputs.append(data_to_device(output, device))
            targets.append(data_to_device(target, device))
        outputs = data_concat(outputs)
        targets = data_concat(targets)
    return outputs, targets

path = os.environ.get("DemoPath")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
seed = 1231
torch.manual_seed(seed=seed)
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Device: {device}")
    SOURCES = ["image", "caption", "label", "history"]
    TARGETS = ["caption", "label"]
    KW_SRC = ["image", "caption", "label", "history"]
    KW_TGT = None
    KW_OUT = None
    # Load the dataset
    INPUT_SIZE = (256, 256)
    MAX_VIEWS = 2
    NUM_LABELS = 114
    NUM_CLASSES = 2
    DATASET_NAME = "NLMCXR"
    MODEL_NAME = "ClsGenInt"
    dataset = NLMCXR_ONE(
    directory=path + "\\NLMCXR\\",
    input_size=INPUT_SIZE,
    view_pos=["AP", "PA", "LATERAL"],
    max_views=MAX_VIEWS,
    sources=SOURCES,
    targets=TARGETS,
    random_transform=True,
)
    test_data = dataset
    VOCAB_SIZE = len(dataset.vocab)
    POSIT_SIZE = dataset.max_len
    COMMENT = "MaxView{}_NumLabel{}_{}History".format(
        MAX_VIEWS, NUM_LABELS, "No" if "history" not in SOURCES else ""
    )
    BACKBONE_NAME = "DenseNet121"
    backbone = torch.hub.load(
        "pytorch/vision:v0.5.0",
        "densenet121",
        weights="DenseNet121_Weights.IMAGENET1K_V1",
        verbose=False,
    )
    FC_FEATURES = 1024

    LR = 3e-5
    WD = 1e-2
    DROPDOUT = 0.1
    NUM_EMBEDS = 256
    FWD_DIM = 256

    NUM_HEADS = 8
    NUM_LAYERS = 1
    cnn = CNN(backbone, BACKBONE_NAME)
    cnn = MVCNN(cnn)
    tnn = TNN(
        embed_dim=NUM_EMBEDS,
        num_heads=NUM_HEADS,
        fwd_dim=FWD_DIM,
        dropout=DROPDOUT,
        num_layers=NUM_LAYERS,
        num_tokens=VOCAB_SIZE,
        num_posits=POSIT_SIZE,
    )

    NUM_HEADS = 1
    NUM_LAYERS = 12
    cls_model = Classifier(
        num_topics=NUM_LABELS,
        num_states=NUM_CLASSES,
        cnn=cnn,
        tnn=tnn,
        fc_features=FC_FEATURES,
        embed_dim=NUM_EMBEDS,
        num_heads=NUM_HEADS,
        dropout=DROPDOUT,
    )
    gen_model = Generator(
        num_tokens=VOCAB_SIZE,
        num_posits=POSIT_SIZE,
        embed_dim=NUM_EMBEDS,
        num_heads=NUM_HEADS,
        fwd_dim=FWD_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPDOUT,
    )
    clsgen_model = Clsgen(cls_model, gen_model, NUM_LABELS, NUM_EMBEDS)
    clsgen_model = nn.DataParallel(clsgen_model).cuda()

    NUM_HEADS = 8
    NUM_LAYERS = 1
    tnn = TNN(
        embed_dim=NUM_EMBEDS,
        num_heads=NUM_HEADS,
        fwd_dim=FWD_DIM,
        dropout=DROPDOUT,
        num_layers=NUM_LAYERS,
        num_tokens=VOCAB_SIZE,
        num_posits=POSIT_SIZE,
    )
    int_model = Classifier(
        num_topics=NUM_LABELS,
        num_states=NUM_CLASSES,
        cnn=None,
        tnn=tnn,
        embed_dim=NUM_EMBEDS,
        num_heads=NUM_HEADS,
        dropout=DROPDOUT,
    )
    int_model = nn.DataParallel(int_model).cuda()
    model = ClsGenInt(
        clsgen_model.module.cpu(), int_model.module.cpu(), freeze_evaluator=True
    )
    criterion = CELossTotalEval(ignore_index=3)
    test_loader = data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    model = nn.DataParallel(model).cuda()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD
    )
    MILESTONES = [25]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)
    checkpoint_path_from = path + "\\checkpoints\\{}_{}_{}_{}.pt".format(
        DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT
    )
    last_epoch, (best_metric, test_metric) = load(
        checkpoint_path_from, model, optimizer, scheduler
    )
    
    txt_test_outputs, txt_test_targets = infer(
        test_loader, model, device="cuda", threshold=0.25
    )
