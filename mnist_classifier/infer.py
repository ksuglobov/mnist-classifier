import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mnist_classifier import logger
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from tqdm import tqdm


def infer(path2model, path2output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_batch_size = 1

    test_dataset = mnist.MNIST(
        root="./test", train=False, transform=ToTensor(), download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = torch.load(path2model)

    all_correct_num = 0
    all_sample_num = 0
    model.eval()

    res = torch.tensor([]).to(device)
    for _, (test_x, test_label) in enumerate(
        tqdm(test_loader, desc="Inference", disable=logger.level > logging.INFO)
    ):
        test_x = test_x.to(device)
        test_label = test_label.to(device)
        predict_y = model(test_x.float()).detach()
        predict_y = torch.argmax(predict_y, dim=-1)
        current_correct_num = predict_y == test_label

        res = torch.cat([res, torch.stack([predict_y, test_label], dim=1)])

        all_correct_num += np.sum(current_correct_num.to("cpu").numpy(), axis=-1)
        all_sample_num += current_correct_num.shape[0]
    acc = all_correct_num / all_sample_num

    logger.info(f"accuracy: {acc:.3f}")

    res = res.to("cpu")

    path2output = str(Path(path2output_dir, "labels.csv"))
    df = pd.DataFrame(res, columns=["pred labels", "true labels"])
    df.to_csv(path2output)
    logger.info(f"Output table (with true and predicted labels) saved to {path2output}")
