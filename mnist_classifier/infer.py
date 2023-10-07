import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor


if __name__ == "__main__":
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cpu"
    test_batch_size = 1

    test_dataset = mnist.MNIST(
        root="./test", train=False, transform=ToTensor(), download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = torch.load("model.pkl")

    all_correct_num = 0
    all_sample_num = 0
    model.eval()

    res = torch.tensor([]).to(device)
    for _, (test_x, test_label) in enumerate(test_loader):
        test_x = test_x.to(device)
        test_label = test_label.to(device)
        predict_y = model(test_x.float()).detach()
        predict_y = torch.argmax(predict_y, dim=-1)
        current_correct_num = predict_y == test_label

        res = torch.cat([res, torch.stack([predict_y, test_label], dim=1)])

        all_correct_num += np.sum(current_correct_num.to("cpu").numpy(), axis=-1)
        all_sample_num += current_correct_num.shape[0]
    acc = all_correct_num / all_sample_num

    print(f"  accuracy: {acc:.3f}", flush=True)

    re = res.to("cpu")
    df = pd.DataFrame(res, columns=["pred labels", "true labels"])
    res_filename = "labels.csv"
    df.to_csv(res_filename)
    print(f"Table with true and predicted labels is written to {res_filename}")
