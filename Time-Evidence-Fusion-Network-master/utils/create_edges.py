import torch

def create_ETT_h1edges(num_nodes):
    # 节点索引从 0 到 num_nodes-1
    edges = [[0,1,2,3],
             [2,3,0,1]]

    # 将边列表转换为 PyTorch 张量
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def create_ETT_m1edges(num_nodes):
    # 连接所有节点到一个中心节点以确保连通性
    edges = []
    for i in range(num_nodes - 1):
        edges.append([i, num_nodes - 1])
    # 将边列表转换为 PyTorch 张量
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index
