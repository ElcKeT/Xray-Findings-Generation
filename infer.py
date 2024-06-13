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
                output = model(image=source[0], history=source[3], threshold=threshold)
            else:
                output = model(image=source[0])

            outputs.append(data_to_device(output, device))
            targets.append(data_to_device(target, device))
        outputs = data_concat(outputs)
        targets = data_concat(targets)
    return outputs, targets


def infer_once(data_loader, model, device="cpu", threshold=None):
    model.eval()
    outputs = []
    targets = []
    with torch.no_grad():
        # prog_bar = tqdm(data_loader, desc="Infer", leave=False)
        for source, target in data_loader:
            source = data_to_device(source, device)
            target = data_to_device(target, device)
            if threshold is not None:
                output = model(
                    image=source[0],
                    history=source[3],
                    threshold=threshold,
                )

            else:
                output = model(image=source[0])
            outputs.append(data_to_device(output, device))
            targets.append(data_to_device(target, device))
            break
        outputs = data_concat(outputs)
        targets = data_concat(targets)
    return outputs, targets


path = os.environ.get("DemoPath")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
seed = 1231
torch.manual_seed(seed=seed)

EPOCHS = 50  # Start overfitting after 20 epochs
BATCH_SIZE = 64
MILESTONES = [25]  # Reduce LR by 10 after reaching milestone epochs

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

    dataset = NLMCXR(
        directory=path
        + "\\NLMCXR\\",
        input_size=INPUT_SIZE,
        view_pos=["AP", "PA", "LATERAL"],
        max_views=MAX_VIEWS,
        sources=SOURCES,
        targets=TARGETS,
        random_transform=True,
    )

    train_data, val_data, test_data = dataset.get_subsets(seed=seed)

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

    train_loader = data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )
    test_loader = data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )
    test_loader_once = data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )
    model = nn.DataParallel(model).cuda()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)
    # print("Total Parameters:", sum(p.numel() for p in model.parameters()))

    last_epoch = -1
    best_metric = 1e9

    checkpoint_path_from = (
        path
        + "\\checkpoints\\{}_{}_{}_{}.pt".format(
            DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT
        )
    )
    last_epoch, (best_metric, test_metric) = load(
        checkpoint_path_from, model, optimizer, scheduler
    )
    """print(
        "Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}".format(
            checkpoint_path_from, last_epoch, best_metric, test_metric
        )
    )"""
    INFER_MODEL = "INFER_ONCE"  # "INFER_ALL" or "INFER_ONCE"
    if INFER_MODEL == "INFER_ALL":
        txt_test_outputs, txt_test_targets = infer(
            test_loader, model, device="cuda", threshold=0.25
        )
    elif INFER_MODEL == "INFER_ONCE":
        txt_test_outputs, txt_test_targets = infer_once(
            test_loader_once, model, device="cuda", threshold=0.25
        )
    else:
        raise ValueError("INFER_MODEL must be either INFER_ALL or INFER_ONCE")
    gen_outputs = txt_test_outputs[0]
    gen_targets = txt_test_targets[0]
    if INFER_MODEL == "INFER_ALL":
        out_file_ref = open(
            path + "\\outputs\\x_{}_{}_{}_{}_Ref.txt".format(
                DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT
            ),
            "w",
        )
        out_file_hyp = open(
            path
            + "\\outputs\\x_{}_{}_{}_{}_Hyp.txt".format(
                DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT
            ),
            "w",
        )
        out_file_lbl = open(
            path
            + "\\outputs\\x_{}_{}_{}_{}_Lbl.txt".format(
                DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT
            ),
            "w",
        )

    for i in range(len(gen_outputs)):
        candidate = ""
        for j in range(len(gen_outputs[i])):
            tok = dataset.vocab.id_to_piece(int(gen_outputs[i, j]))
            if tok == "</s>":
                break  # Manually stop generating token after </s> is reached
            elif tok == "<s>":
                continue
            elif tok == "▁":  # space
                if len(candidate) and candidate[-1] != " ":
                    candidate += " "
            elif tok in [",", ".", "-", ":"]:  # or not tok.isalpha():
                if len(candidate) and candidate[-1] != " ":
                    candidate += " " + tok + " "
                else:
                    candidate += tok + " "
            else:  # letter
                candidate += tok

        reference = ""
        for j in range(len(gen_targets[i])):
            tok = dataset.vocab.id_to_piece(int(gen_targets[i, j]))
            if tok == "</s>":
                break
            elif tok == "<s>":
                continue
            elif tok == "▁":  # space
                if len(reference) and reference[-1] != " ":
                    reference += " "
            elif tok in [",", ".", "-", ":"]:  # or not tok.isalpha():
                if len(reference) and reference[-1] != " ":
                    reference += " " + tok + " "
                else:
                    reference += tok + " "
            else:  # letter
                reference += tok

        if INFER_MODEL == "INFER_ALL":
            out_file_hyp.write(candidate + "\n")
            out_file_ref.write(reference + "\n")
            for i in tqdm(range(len(test_data))):
                target = test_data[i][1]  # caption, label
            out_file_lbl.write(" ".join(map(str, target[1])) + "\n")

        elif INFER_MODEL == "INFER_ONCE":
            for i in range(
                len(
                    test_loader_once.dataset.file_report[
                        test_loader_once.dataset.file_list[0]
                    ]["image"]
                )
            ):
                print(
                    #"file_name[%d]:%s"
                    "%s"
                    % (
                        test_loader_once.dataset.file_report[
                            test_loader_once.dataset.file_list[0]
                        ]["image"][i]
                    ),
                )
            print("hpy:" + candidate)
            print("ref:" + reference)
            target = test_data[i][1]
            # print("lbl:" + " ".join(map(str, target[1])) + "\n")

            from nlgeval import compute_individual_metrics

            metrics_dict = compute_individual_metrics(
                reference, candidate, no_skipthoughts=True, no_glove=True
            )
            for metric, score in metrics_dict.items():
                if str(score) != "0.0":
                    print("%s: %s" % (metric, score))
            print("\n")

            file_name = test_loader_once.dataset.file_report[
                test_loader_once.dataset.file_list[0]
            ]["image"][0]
            xml_name = test_loader_once.dataset.file_list[0]
            newjson = {}
            newjson[xml_name] = test_loader_once.dataset.file_report[xml_name]
            with open(
                path
                + "\\NLMCXR\\testone.json",
                "w",
            ) as file:
                json.dump(newjson, file)
            newcaption = {}
            newcaption[file_name + ".png"] = candidate
            with open(
                path
                + "\\NLMCXR\\testcaption.json",
                "w",
            ) as file:
                json.dump(newcaption, file)
