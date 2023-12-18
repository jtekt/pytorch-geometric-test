import torch
import torch.nn.functional as F

from torch_geometric.nn.conv import PointNetConv, GraphConv
from torch_geometric.nn import MLP, global_max_pool
from torch_geometric.data import Data


class GraphNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphNetwork, self).__init__()
        self.conv1 = PointNetConv(
            local_nn = MLP([in_channels, hidden_channels, hidden_channels]),
            global_nn  = MLP([hidden_channels, hidden_channels])
        )
        self.dout1 = torch.nn.Dropout(p=0.2)
        self.dout2 = torch.nn.Dropout(p=0.2)
        self.conv2 = GraphConv(
            hidden_channels,
            hidden_channels,
        )
        self.conv3 = GraphConv(
            hidden_channels,
            out_channels,
        )
        self.pool = global_max_pool

    def forward(self, d) -> torch.tensor:
        
        edge = d.edge_index.to(self.device)
        x1 = self.conv1(d.x.to(self.device), d.pos.to(self.device), edge)
        x1 = F.leaky_relu(x1)
        x1 = self.dout1(x1)
        x2 = self.conv2(x1, edge)
        x2 = F.leaky_relu(x2)
        x2 = self.dout2(x2)
        x3 = self.conv3(x2, edge)
        dbatch = [0] if d.batch is None else d.batch
        batch = torch.tensor(dbatch, dtype=torch.long).to(self.device)
        x4 = self.pool(x3, batch)

        return x4

    def set_device(self, device: torch.device):
        self.to(device)
        self.device = device


def test_encoding_spacial():

    # (v, f)
    x = torch.tensor(
        [
            [0],
            [1],
            [1]
        ],
        dtype=torch.float
    )

    # (v, dim) = 3, 2
    pos = torch.tensor(
        [
            [1, 2],
            [2, 2],
            [3, 3]
        ],
        dtype=torch.float
    )
    # (2, v) = 2, 3
    edge_index = torch.tensor(
        [
            [0, 1],
            [0, 2],
            # [1, 0],
            # [2, 0],
        ],
        dtype=torch.long
    ).T

    
    print("x: ", x.shape)
    print("edge_index: ", edge_index.shape)
    print("pos: ", pos.shape)
    d = Data(x=x, pos=pos, edge_index=edge_index)
    print(d)
    model = GraphNetwork(2+1, 20, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = model(d)
    print("ouput:", output.shape)

    loss = F.nll_loss(output, torch.tensor([0]))
    print("loss:", loss)
    
    # msg: torch.Size([5, 3]) x_j torch.Size([5, 1])
    # msg-concat torch.Size([5, 4])
    # msg-local torch.Size([5, 20])
    # x1 torch.Size([3, 20])
    # x2 torch.Size([3, 20])
    # x3 torch.Size([3, 2])
    
    # pos_j: tensor([[1., 2.],
    #     [1., 2.],
    #     [1., 2.],
    #     [2., 2.],
    #     [3., 3.]])
    # pos_i: tensor([[2., 2.],
    #         [3., 3.],
    #         [1., 2.],
    #         [2., 2.],
    #         [3., 3.]])
    # x_j: tensor([[0.],
    #         [0.],
    #         [0.],
    #         [1.],
    #         [1.]])

if __name__ == '__main__':
    

    test_encoding_spacial()