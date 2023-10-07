import numpy as np
import torch
from model import Model
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor


if __name__ == "__main__":
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cpu"
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
    all_epoch = 5
    prev_acc = 0
    best_acc = None
    best_model = None
    for current_epoch in range(all_epoch):
        print(f"Epoch {current_epoch + 1}/{all_epoch}:")
        model.train()
        for _, (train_x, train_label) in enumerate(train_loader):
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

        print(f"  accuracy: {acc:.3f}", flush=True)

        if best_acc is None or acc > best_acc:
            best_acc = acc
            best_model = model

        if np.abs(acc - prev_acc) < 1e-4:
            break
        prev_acc = acc
    # if not os.path.isdir("models"):
    #         os.mkdir("models")
    torch.save(best_model, "model.pkl")
    print("Model finished training")
