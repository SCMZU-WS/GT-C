import torch
from torch_geometric.utils import degree, dropout_adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch import nn
from typing import Any, List, Callable, Tuple

class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')
        
        self.emb_dim = emb_dim
        # bias=False:keep padding to 0
        self.linear = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.root_emb = torch.nn.Embedding(2, emb_dim, padding_idx=0)
        self.edge_encoder = torch.nn.Linear(7, emb_dim,bias=False)

    def forward(self, x, edge_index, edge_attr, root):
        x = self.linear(x.view(-1, self.emb_dim))
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb(root)) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class MSA_BLOCK(nn.Module):
    def __init__(
        self,
        edge_dim: int=7,
        head_num: int=8,
        feature_dim: int=128,
        atte_drop_rate: float=0.,
        msa_drop_rate: float=0.,
        qkv_drop: float=0.5,
    ):
        super(MSA_BLOCK, self).__init__()
        self.head_num = head_num
        self.one_head_dim = feature_dim // head_num
        self.scale = self.one_head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.q = nn.Linear(feature_dim, self.one_head_dim*head_num, bias=False)
        self.k = nn.Linear(feature_dim, self.one_head_dim*head_num, bias=False)
        self.v = nn.Linear(feature_dim, self.one_head_dim*head_num, bias=False)
        self.qkv_drop = qkv_drop
        self.attn_drop = nn.Dropout(atte_drop_rate)
        self.msa_dropout = nn.Dropout(msa_drop_rate)
        self.merge_head = nn.Linear(self.one_head_dim*head_num, feature_dim)
        
        
    def forward(self, x, bias=None):
        
        # x dim:[batch, node_num, feature_dim]
        batch_num, node_num, feature_dim = x.shape[:]
        
        # [batch, node_num, one_head_dim*head_num] 
        # -> [batch, head_num, node_num, one_head_dim]
        q = F.dropout(self.q(x).view(batch_num,-1,self.head_num,self.one_head_dim).permute(0,2,1,3), self.qkv_drop, training=self.training)
        k = F.dropout(self.k(x).view(batch_num,-1,self.head_num,self.one_head_dim).permute(0,2,1,3), self.qkv_drop, training=self.training)
        v = F.dropout(self.v(x).view(batch_num,-1,self.head_num,self.one_head_dim).permute(0,2,1,3), self.qkv_drop, training=self.training)
    
        q = q * self.scale
        qk_atten = q @ k.transpose(-2,-1)
        if bias is not None:
            qk_atten = qk_atten + bias.permute(0,3,1,2)
        qk_atten = self.softmax(qk_atten)
            
        # [batch, head_num, node_num, one_head_dim]
        # -> [batch, node_num, head_num, one_head_dim]
        # -> [batch, node_num, head_num*one_head_dim]
        qkv = (self.attn_drop(qk_atten) @ v).transpose(1,2).reshape(batch_num,node_num,-1)
        
        x = self.merge_head(qkv)
        
        return self.msa_dropout(x)


class Encoder(nn.Module):
    def __init__(
        self,
        edge_dim: int=7,
        head_num: int=8,
        feature_dim: int=128,
        atte_drop_rate: float=0.,
        drop_rate: float=0.,
        qkv_drop: float=0.5,
        infla_dim: int=1,
        device: str='cpu',
    ):
        super(Encoder, self).__init__()
        self.layer_norm_input = nn.LayerNorm(feature_dim)
        self.embedding_mcl = nn.Linear(infla_dim, head_num, bias=False)
        self.msa = MSA_BLOCK(
            edge_dim=edge_dim,
            head_num=head_num,
            feature_dim=feature_dim,
            atte_drop_rate=atte_drop_rate,
            msa_drop_rate=drop_rate,
            qkv_drop=qkv_drop,
        )
        
        self.layer_norm_ffn = nn.LayerNorm(feature_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*2),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(feature_dim*2, feature_dim)
        )
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self, x, bias, mcl_mat=None):
        n_batch, n_nodes, _ = x.shape
        if mcl_mat is not None:
            # mcl bias
            mcl_bias = self.embedding_mcl(mcl_mat)
            bias = bias + mcl_bias
            
        res = x
        # post norm
        x = self.layer_norm_input(self.msa(x, bias) + res)
        
        res = x
        x = self.ffn(x)
        x = res + self.dropout(x).view(n_batch, n_nodes, -1)
        
        return self.layer_norm_ffn(x)
    
class GC_T(nn.Module):
    def __init__(
        self,
        max_num_nodes: int=300,
        edge_dim: int=7,
        head_num: int=8,
        feature_dim: int=128,
        input_drop_rate: float=0.2,
        atte_drop_rate: float=0.1,
        drop_rate: float=0.1,
        layer_drop: float=0.5,
        qkv_drop: float=0.5,
        n_layers: int=9,
        class_num: int=37,
        infla: list=[6.0],
        device="cpu"
    ):
        super(GC_T, self).__init__()
        self.half_layer = int(n_layers // 2)
        self.n_layers = n_layers
        self.edge_dim = edge_dim
        self.head_num = head_num
        self.inflation = infla
        self.device = device
        self.max_num_nodes = max_num_nodes
        self.input_dropout = nn.Dropout(input_drop_rate)
        self.layer_drop = layer_drop
        self.gcn = nn.ModuleList()
        self.gcn_norm = nn.ModuleList()
        self.trans = nn.ModuleList()
        for i in range(n_layers):
            self.gcn.append(GCNConv(feature_dim))
            self.gcn_norm.append(nn.LayerNorm(feature_dim))
            self.trans.append(
                Encoder(
                    edge_dim=edge_dim,
                    head_num=head_num,
                    feature_dim=feature_dim,
                    atte_drop_rate=atte_drop_rate,
                    drop_rate=drop_rate,
                    qkv_drop=qkv_drop,
                    infla_dim=len(self.inflation),
                    device=device,
                )
            )
        self.final_ln = nn.LayerNorm(feature_dim)

        self.apply(
            lambda module: GC_T._init_layer_params(
                module,
                n_layers=n_layers
            )
        )
        
    @staticmethod
    def _init_layer_params(module, n_layers):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02 * (n_layers ** -0.5))
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            
    @staticmethod
    def _get_mcl(adj, inflation, num: int=6):
        mcl_mat = adj/(adj.sum(dim=-1, keepdim=True)+1e-6)
        result = mcl_mat
        # expansion and inflation operation
        for i in range(num):
            result = mcl_mat @ result
            if (i+1)%3==0:
                result = torch.pow(result, inflation)
            result = result/(result.sum(dim=-1, keepdim=True)+1e-6)
        return result
    
    def forward(self, input_x, batch_data, edge_index, edge_attr):
        batch = batch_data.batch
        n_batch = int(batch.max())+1
        n_nodes = self.max_num_nodes
        batch_index = batch.view(n_batch, -1)
        
        # input feature
        # [batch * n_nodes, input_feature] -> [batch, n_nodes, input_feature]
        node_feature = input_x.view(n_batch, n_nodes, -1)
        
        # edge mask
        edge_mask = torch.div(edge_index[0], n_nodes, rounding_mode='floor') # for edge index
        source_list = (edge_index[0]%n_nodes).tolist()
        target_list = (edge_index[1]%n_nodes).tolist()
        
        # adj and padding bias
        adj_mat = torch.zeros(n_batch,n_nodes,n_nodes, dtype=torch.bool).to(self.device)
        pad_mat = torch.ones(n_batch,n_nodes,n_nodes, dtype=torch.bool).to(self.device)
        adj_mat[edge_mask.tolist(),source_list,target_list] = True
        for i, num in enumerate(batch_data.n): # add self loop
            adj_mat[i, range(num), range(num)] = True
            pad_mat[i, :num,:num] = False
        pad_bias = pad_mat * -1024
        pad_bias = pad_bias.unsqueeze(-1).repeat(1, 1, 1, self.head_num)
            
        # compute mcl mat
        mcl_mat = torch.stack([
            GC_T._get_mcl(adj_mat,inflation=i).to(self.device) for i in self.inflation
        ], dim=-1)
        
        output = self.input_dropout(node_feature)
        for i in range(self.n_layers):
            res = output
            output = self.gcn[i](output, edge_index, edge_attr, batch_data.x)
            output = self.gcn_norm[i](output)
            output = F.dropout(F.relu(output), 0.3, training=self.training).view(n_batch, n_nodes, -1) + res
            p = torch.rand(1).item()
            if self.training and p<=self.layer_drop:
                # drop layer
                continue
            output = self.trans[i](output, pad_bias.float(), mcl_mat)

        return self.final_ln(output)
    

class NET(nn.Module):
    def __init__(
        self,
        head_num: int=8,
        input_drop_rate: float=0.2,
        atte_drop_rate: float=0.1,
        layer_drop: float=0.5,
        qkv_drop: float=0.5,
        n_layers: int=3,
        max_num_nodes: int=300,
        edge_dim: int=7,
        feature_dim: int=128,
        class_num: int=37,
        infla: list=[6.0],
        device="cpu"
    ):
        super(NET, self).__init__()
        self.node_encoder = torch.nn.Embedding(2, feature_dim, padding_idx=0)
        self.node_encoder.weight.data.normal_(mean=0.0, std=0.02)
        self.gct = GC_T(
            max_num_nodes=max_num_nodes,
            edge_dim=edge_dim,
            head_num=head_num,
            feature_dim=feature_dim,
            input_drop_rate=input_drop_rate,
            atte_drop_rate=atte_drop_rate,
            n_layers=n_layers,
            class_num=class_num,
            infla=infla,
            device=device
        )
        self.feature_dim= feature_dim
        # graph pooling
        self.pool = global_mean_pool
        self.graph_pred_linear = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, class_num),
        )

        
    def forward(self,batch_data):
        x, edge_index, edge_attr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch
        x = self.node_encoder(x)
        x = self.gct(x,batch_data,edge_index, edge_attr).view(-1,self.feature_dim)
        node_mask = batch_data.x.bool() # for padding
        h_graph = self.pool(x[node_mask], batch[node_mask])
        
        return self.graph_pred_linear(h_graph)
