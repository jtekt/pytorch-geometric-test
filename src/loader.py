import numpy as np
import torch
from scipy import stats
from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader


# generate 1000 samples from the bivariate normal distribution
def generate_normal(
    mean: list[float],
    cov: list[list[float]],
    size: int
) -> np.ndarray:
    return np.random.multivariate_normal(mean, cov, size=size)


def generate_normal_bernoulliASmean(
    mean: list[float],
    cov: list[list[float]],
    size: int,
    binomial_maen: list[float],
    p_bernoulli: float
) -> np.ndarray:

    bernoulli_sample = stats.bernoulli.rvs(p=p_bernoulli, size=size)[:, np.newaxis]
    binomial_maen_tile = np.tile(binomial_maen, (size,1))
    ret = generate_normal(mean, cov, size) + bernoulli_sample * binomial_maen_tile
    return  ret


# data_list = [Data(...), ..., Data(...)]
# loader = DataLoader(data_list, batch_size=32)


def make_dataset_2d_each(
    mean,
    cov,
    size,
    noise_cov,
    binomial_maen,
    p_bernoulli: float,
    p_noise: float
) -> tuple:
    features = stats.bernoulli.rvs(p=p_noise, size=size)
    noise = generate_normal(
        [0, 0],
        noise_cov,
        size
    )
    # normal = generate_normal(mean, cov, size)
    data = generate_normal_bernoulliASmean(
        mean,
        cov,
        size,
        binomial_maen,
        p_bernoulli
    )

    data_noise = data + features[:, np.newaxis] * noise
    # ret = [ Data(x=f, pos=d) for d, f in zip(data_noise, features)]
    features = features + 1
    return (data_noise, features)


def make_normal_bias(
    mean,
    cov,
    biasList,
    noise_cov,
    p_noise
) -> tuple:
    size = len(biasList)
    features = stats.bernoulli.rvs(p=p_noise, size=size)
    noise = generate_normal(
        [0, 0],
        noise_cov,
        size
    )
    noise = features[:, np.newaxis] * noise
    data = generate_normal(mean, cov, size)
    
    ret = data + biasList + noise
    features = features + 1
    # features[:] = 1
    # print(features.shape, type(features))
    return (ret, features)
    

def make_dataset_graph1(
    meanList: list[list[float]],
    covList: list[list[list[float]]],
    size: int,
    noise_cov: list[list[float]],
    binomial_maen: list[float],
    p_bernoulli: float,
    p_noise: float,
    y: int
) -> list[Data]:  
    
    dataList = list()
    featuresList = list()
    # (2, num_edges)
    edge_index = torch.tensor(
        [
            [0, 1],
            [0, 2]
        ],
        dtype=torch.long
    ).T
    
    data_org, features_org = make_dataset_2d_each(
        meanList[0],
        covList[0],
        size,
        noise_cov,
        binomial_maen,
        p_bernoulli,
        p_noise
    )
    dataList.append(data_org)
    featuresList.append(features_org)
    
    # 
    for mean, cov in zip(meanList[1::], covList[1::]):
        data, features = make_normal_bias(
            mean,
            cov,
            data_org,
            noise_cov,
            p_noise
        )
        dataList.append(data)
        featuresList.append(features)

    dataList = np.hstack(dataList)
    featuresList = np.array(featuresList).T[:, :, np.newaxis]

    ret = [
        Data(
            x=torch.tensor(f),
            pos=torch.tensor(
                d, dtype=torch.float
            ).reshape((3,2)),
            y=torch.tensor([y], dtype=torch.long),
            edge_index=edge_index
        ) for d, f in zip(dataList, featuresList)
    ]

    return ret


def make_dataset_graph2(
    meanList,
    covList,
    size: int,
    noise_cov: list[list[float]],
    binomial_maen: list[float],
    p_bernoulli: float,
    p_noise: float,
    y: int
) -> list[Data]:
    dataList = list()
    featuresList = list()
    # (2, num_edges)
    edge_index = torch.tensor(
        [
            [0, 2]
        ],
        dtype=torch.long
    ).T

    data_org, features_org = make_dataset_2d_each(
        meanList[0],
        covList[0],
        size,
        noise_cov,
        binomial_maen,
        p_bernoulli,
        p_noise
    )
    dataList.append(data_org)
    featuresList.append(features_org)
    data, features = make_normal_bias(
        meanList[0],
        covList[0],
        np.zeros_like(data_org),
        noise_cov,
        p_noise
    )
    dataList.append(data)
    featuresList.append(features)
    # 
    for mean, cov in zip(meanList[2::], covList[2::]):
        data, features = make_normal_bias(
            mean,
            cov,
            data_org,
            noise_cov,
            p_noise
        )
        dataList.append(data)
        featuresList.append(features)

    dataList = np.hstack(dataList)
    featuresList = np.array(featuresList).T[:, :, np.newaxis]

    ret = [
        Data(
            x=torch.tensor(f),
            pos=torch.tensor(
                d, dtype=torch.float
            ).reshape((3,2)),
            y=torch.tensor([y], dtype=torch.long),
            edge_index=edge_index
        ) for d, f in zip(dataList, featuresList)
    ]

    return ret

def generation():

    mean = [[0, 0], [1, 1], [2, 2]]
    cov = [
        [
            [0.2, 0],
            [0, 0.2]
        ],
        [
            [0.2, 0],
            [0, 0.2]
        ],
        [
            [0.2, 0],
            [0, 0.2]
        ]
    ]
    noise_cov1 = [
        [1, 0],
        [0, 1]
    ]
    size1_1 = 300
    size1_2 = 100
    binomial_maen = [1, 1]
    p_bernoulli = 0.5
    p_noise = 0.4
    dataset1_1 = make_dataset_graph1(
        mean,
        cov,
        size1_1,
        noise_cov1,
        binomial_maen,
        p_bernoulli,
        p_noise,
        0
    )
    dataset1_2 = make_dataset_graph2(
        mean,
        cov,
        size1_2,
        noise_cov1,
        binomial_maen,
        p_bernoulli,
        p_noise,
        0
    )

    mean2 = [[0.1, 0.3], [1.2, 0.3], [1.0, 1.1]]
    cov2 = [
        [
            [0.2, 0],
            [0, 0.2]
        ],
        [
            [0.2, 0],
            [0, 0.2]
        ],
        [
            [0.2, 0],
            [0, 0.2]
        ]
    ]
    size2_1 = 300
    size2_2 = 300
    noise_cov2 = [
        [1, 0],
        [0, 1]
    ]
    binomial_maen2 = [1, 2]
    p_bernoulli2 = 0.6
    p_noise2 = 0.4
    dataset2_1 = make_dataset_graph2(
        mean2,
        cov2,
        size2_1,
        noise_cov2,
        binomial_maen2,
        p_bernoulli2,
        p_noise2,
        1
    )
    dataset2_2 = make_dataset_graph2(
        mean2,
        cov2,
        size2_2,
        noise_cov2,
        binomial_maen2,
        p_bernoulli2,
        p_noise2,
        1
    )
    # print(dataset1[0])
    # print(dataset2[0])
    # print(len(dataset2))

    dataset = dataset1_1 + dataset2_1 + dataset1_2 + dataset2_2
    # print(len(dataset))
    return dataset



def randomsort_split_trainval(
    dataset: list[Data],
    train_ratio: float = 0.7
) -> tuple[list[Data], list[Data]]:
    import copy
    import random

    size = len(dataset)
    train_size = int(size * train_ratio)
    
    dataset_random = copy.deepcopy(dataset)
    random.shuffle(dataset_random)
    data_train = dataset_random[0:train_size]
    data_val = dataset_random[train_size::]

    return (data_train, data_val)


if __name__ == '__main__':

    dataset = generation()

    train, val = randomsort_split_trainval(dataset)