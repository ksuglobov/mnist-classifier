import logging
from pathlib import Path

import numpy as np
import torch
from mnist_classifier import logger
from mnist_classifier.model import Model
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from tqdm import tqdm


def train(n_epochs, path2model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_batch_size = 64
    val_batch_size = 64

    mnist_dataset = mnist.MNIST(
        root="./train", train=True, transform=ToTensor(), download=True
    )
    train_size = 0.8

    mnist_len = len(mnist_dataset)
    train_len = int(train_size * mnist_len)
    val_len = mnist_len - train_len

    train_dataset, val_dataset = random_split(mnist_dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    model = Model().to(device)
    sgd = SGD(model.parameters(), lr=1e-1)
    loss_fn = CrossEntropyLoss()
    prev_acc = 0
    best_acc = None
    best_model = None
    for current_epoch in range(n_epochs):
        logger.info(f"Epoch {current_epoch + 1}/{n_epochs}")
        model.train()
        for _, (train_x, train_label) in enumerate(
            tqdm(train_loader, desc="Training epoch", disable=logger.level > logging.INFO)
        ):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            sgd.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()

        for _, (val_x, val_label) in enumerate(val_loader):
            val_x = val_x.to(device)
            val_label = val_label.to(device)
            predict_y = model(val_x.float()).detach()
            predict_y = torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == val_label
            all_correct_num += np.sum(current_correct_num.to("cpu").numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num

        logger.info(f"accuracy: {acc:.3f}")

        if best_acc is None or acc > best_acc:
            best_acc = acc
            best_model = model

        if np.abs(acc - prev_acc) < 1e-4:
            break
        prev_acc = acc

    path2model = str(Path(path2model_dir, "model.pkl"))
    torch.save(best_model, path2model)
    logger.info(f"Trained model with best accuracy {best_acc:.3f} saved to {path2model}")
