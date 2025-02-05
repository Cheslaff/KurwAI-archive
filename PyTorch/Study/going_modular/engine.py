import torch
from torchmetrics import Accuracy

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    accuracy_fn = Accuracy(task="multiclass", num_classes=3)
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        logits = model(X)
        pred = torch.softmax(logits, dim=1).argmax(dim=1)

        loss = loss_fn(logits, y)
        accuracy = accuracy_fn(pred, y)

        train_loss += loss
        train_acc += accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    accuracy_fn = Accuracy("multiclass", num_classes=3)
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            predictions = torch.softmax(logits, dim=1).argmax(dim=1)
            test_loss += loss_fn(logits, y)
            test_acc += accuracy_fn(predictions, y)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          verbose: bool=True):
    
    model.to(device)
    
    metrics = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model,
                                           train_dataloader,
                                           loss_fn,
                                           optimizer,
                                           device)
        test_loss, test_acc = test_step(model,
                                        test_dataloader,
                                        loss_fn,
                                        device)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["test_loss"].append(test_loss)
        metrics["test_acc"].append(test_acc)

        if verbose:
            if epoch % 10:
                print(f"Epoch: {epoch} Train loss: {train_loss} \
                        Train acc: {train_acc} \
                        Test loss: {test_loss} \
                        Test acc: {test_acc}")

    return metrics

