import logging
from networks.layers.coll_conv import CollConv
import torch
import torch.nn as nn
from torch.nn import Sequential
from networks.layers.util import MLP, Linear_trans


class PseudoTilinGNN(torch.nn.Module):
    def __init__(self,
                 network_depth=10,
                 network_width=32,
                 output_dim=1,
                 raw_node_features_dim=1):
        super(PseudoTilinGNN, self).__init__()

        self.network_depth = network_depth
        self.network_width = network_width
        self.residual_skip_num = 2

        self.collision_branch_feature_dims = [self.network_width
                                              ] * (self.network_depth + 1)

        # MLP processes the raw features
        self.init_node_feature_trans = MLP(
            in_dim=raw_node_features_dim,
            out_dim=self.network_width,
            hidden_layer_dims=[self.network_width],
            activation=torch.nn.LeakyReLU(),
            batch_norm=True)

        # graph convolution layers
        self.collision_branch_layers = nn.ModuleList([
            CollConv(
                node_feature_in_dim=self.collision_branch_feature_dims[i],
                node_feature_out_dim=self.collision_branch_feature_dims[i + 1])
            for i in range(len(self.collision_branch_feature_dims) - 1)
        ])

        # output layers
        self.final_mlp = Sequential(
            # MLP(in_dim=sum(self.collision_branch_feature_dims),
            MLP(in_dim=self.collision_branch_feature_dims[-1],
                out_dim=self.network_width,
                hidden_layer_dims=[256, 128, 64],
                activation=torch.nn.LeakyReLU()),
            Linear_trans(self.network_width,
                         output_dim,
                         activation=torch.nn.Sigmoid(),
                         batch_norm=False))

    def forward(self, x, col_e_idx, col_e_features=None):
        # logging.debug("input network:shape: {}".format(x.shape))

        # MLP process the raw features
        collision_branch_feature = self.init_node_feature_trans(x)

        torch.cuda.empty_cache()
        # main procedure
        middle_features = [collision_branch_feature]
        for i in range(self.network_depth):
            coll_layer = self.collision_branch_layers[i]
            collision_branch_feature, *_ = coll_layer(collision_branch_feature,
                                                      col_e_idx)

            # residual connection
            prev_layer_num = i - self.residual_skip_num
            if prev_layer_num >= 0:
                collision_branch_feature += middle_features[prev_layer_num]

            middle_features.append(collision_branch_feature)

        # output layers
        # skip_connec_features = torch.cat(middle_features, 1)
        skip_connec_features = collision_branch_feature

        torch.cuda.empty_cache()
        node_features = self.final_mlp(skip_connec_features)

        # logging.debug("output network:shape:{}".format(x.shape))

        return node_features
