import torch
import torch.nn.functional as F
import numpy as np


from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from src import loader
from src.mynetwork import GraphNetwork
from src.logconf import mylogger
logger = mylogger(__name__)


def train_epoch(
    model: torch.nn.Module,
    dataloader_train: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:

    model.train()
    total_loss = 0
    correct_labels_sum = 0
    for data in dataloader_train:
        optimizer.zero_grad()
        out = model.forward(data)
        loss_train = F.nll_loss(F.log_softmax(out, dim=1), data.y.to(device))

        pred = F.softmax(out, dim=1).detach().cpu().max(dim=1)[1]
        correct_labels_sum += pred.eq(data.y.cpu()).sum().item()
        loss_train.backward()
        total_loss += loss_train.item() * data.num_graphs
        optimizer.step()
    
    return (
        total_loss / len(dataloader_train),
        correct_labels_sum / len(dataloader_train.dataset)
    )


def test_epoch(
    model: torch.nn.Module,
    dataloader_val: DataLoader,
    device: torch.device
) -> tuple[float, float]:

    model.eval()

    total_loss = 0
    correct_labels_sum = 0
    for data in dataloader_val:
        
        with torch.no_grad():
            out = model.forward(data)
            loss_val = F.nll_loss(F.log_softmax(out, dim=1), data.y.to(device))

        total_loss += loss_val.item() * data.num_graphs
        pred = F.softmax(out, dim=1).detach().cpu().max(dim=1)[1]
        correct_labels_sum += pred.eq(data.y.cpu()).sum().item()

    return (
        total_loss / len(dataloader_val),
        correct_labels_sum / len(dataloader_val.dataset)
    )


def training(
    lr: float,
    epochs: int,
    batch_size: int,
    data_trainList: list[Data],
    data_valList: list[Data],
    device: torch.device
) -> dict:

    logger.info("training...")

    model = GraphNetwork(2+1, 50, 2)
    model.set_device(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    epochList = list()
    loss_trainList = list()
    loss_valList = list()
    acc_trainList = list()
    acc_valList = list()
    for epoch in range(1, epochs+1):
        dataloader_train = DataLoader(data_trainList, batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(data_valList, batch_size=batch_size)

        loss_train, acc_train = train_epoch(model, dataloader_train, optimizer, device)
        loss_val, acc_val = test_epoch(model, dataloader_val, device)
        # test_acc = 0.0
        logger.info(
            f'Epoch {epoch:03d}, Loss: {loss_train:.4f}, AccTrain: {acc_train:.4f}  LossVal: {loss_val:.4f} AccVal: {acc_val:.4f}'
        )
        scheduler.step()
        epochList.append(epoch)
        loss_trainList.append(loss_train)
        loss_valList.append(loss_val)
        acc_trainList.append(acc_train)
        acc_valList.append(acc_val)

    logger.info("...finished")
    return dict(
        epoch=epochList,
        loss_train=loss_trainList,
        loss_val=loss_valList,
        acc_train=acc_trainList,
        acc_val=acc_valList
    )


if __name__ == '__main__':

    import argparse
    # from src import visualize
    parser = argparse.ArgumentParser(
        description='train graph convnet.'
    )
    parser.add_argument(
        '--epochs', '-E', type=int, default=100, help=''
    )
    parser.add_argument(
        '--batch_size', '-B', type=int, default=10, help=''
    )
    parser.add_argument(
        '--lr', '-L', type=float, default=1e-4, help=''
    )
    parser.add_argument(
        '--gpu', '-GPU', type=int, default=-1, help=''
    )
    args = parser.parse_args()
    
    dataset = loader.generation()
    data_train, data_val = loader.randomsort_split_trainval(dataset)

    # path_export = './data'
    # if not os.path.exists(path_export):
    #     os.mkdir(path_export)
    # visualize.dataset(dataset, f"{path_export}/all.jpg")
    # visualize.dataset(data_train, f"{path_export}/train.jpg")
    # visualize.dataset(data_val, f"{path_export}/val.jpg")

    device = torch.device(
        f"cuda:{str(args.gpu)}" if args.gpu >= 0 else 'cpu'
)
    result = training(
        args.lr,
        args.epochs,
        args.batch_size,
        data_train, data_val, device
    )
    
    # visualize.training_process(result, f"{path_export}/process.jpg")