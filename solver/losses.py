import torch
import numpy as np
import time
import math
import itertools
import traceback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 1e-7

class Losses:
    # evaluate loss for a given data set
    @staticmethod
    def cal_avg_loss(network, data_set_loader):
        network.eval()
        # losses = []
        area_losses = []
        collision_losses = []
        # avg_collision_probs_list = []
        # filled_area_list = []
        # avg_align_length_list = []

        for batch in data_set_loader:  
            data = batch.to(device)
            probs = network(x = data.x,
                                col_e_idx = data.edge_index)

            if probs is None:
                continue
            
            loss, La, Lo = Losses.calculate_unsupervised_loss(probs, data.x, data.edge_index)
            area_losses.append(La)
            collision_losses.append(Lo)

            '''
            loss, min_index, _ = Losses.calculate_unsupervised_loss(probs, data.x, data.collide_edge_index,
                                                                    adj_edges_index=data.edge_index,
                                                                    adj_edge_features=data.edge_features)

            losses.append(loss.detach().cpu().numpy())
            '''
        
        area_losses = torch.stack(area_losses)
        collision_losses = torch.stack(collision_losses)
        losses = area_losses * collision_losses

        return torch.mean(losses), torch.mean(area_losses), torch.mean(
            collision_losses)

        


    @staticmethod
    def calculate_unsupervised_loss(probs, node_feature, collide_edge_index):
        # start time
        '''
        start_time = time.time()
        N = probs.shape[0]  # number of nodes
        M = probs.shape[1]  # number of output features
        E_col = collide_edge_index.shape[1] if len(collide_edge_index) > 0 else 0  # to handle corner cases when no collision edge exist
        '''
        losses = []
        # weight a
        AVG_AREA_WEIGHT     = 1.0
        # weight o
        COLLISION_WEIGHT    = 1/math.log(1+1e-1) # 10.0
        
        # Loss a
        avg_tile_area = torch.clamp(torch.mean(probs), min=eps)
        loss_ave_area = torch.log(avg_tile_area)
        La = 1 - AVG_AREA_WEIGHT * loss_ave_area   

        # Loss o
        probs = torch.squeeze(probs)
        if len(collide_edge_index) == 0:
            return 0.0
        prob_i = torch.gather(probs, dim=0, index=collide_edge_index[0])
        prob_k = torch.gather(probs, dim=0, index=collide_edge_index[1])
        Lo = 1 - COLLISION_WEIGHT * torch.mean(
            torch.log(1 - torch.clamp(
                prob_i * prob_k, min=eps, max=1 - eps)))

        loss = La * Lo  

        return loss, La, Lo

        '''
        for sol in range(M):
            solution_prob = probs[:, sol]
            ########### average node area loss (La)
            avg_tile_area = torch.clamp(torch.mean(node_feature[:, -1] * solution_prob), min=eps)
            loss_ave_area = torch.log(avg_tile_area)

            ########### collision feasibility loss (Lo)
            # has collision edge
            if E_col > 0:
                first_index = collide_edge_index[0, :]
                first_prob = torch.gather(solution_prob, dim=0, index=first_index)
                second_index = collide_edge_index[1, :]
                second_prob = torch.gather(solution_prob, dim=0, index=second_index)

                prob_product = torch.clamp(first_prob * second_prob, min=eps, max=1 - eps)
                loss_per_edge = torch.log(1 - prob_product)
                loss_per_edge = loss_per_edge.view(-1)
                loss_feasibility = torch.sum(loss_per_edge) / E_col
            # no collision edge    
            else:
                loss_feasibility = 0.0


            assert loss_feasibility <= 0
            assert loss_ave_area <= 0

            loss = (1 - AVG_AREA_WEIGHT     * loss_ave_area   ) * \
                   (1 - COLLISION_WEIGHT    * loss_feasibility) 
            assert loss >= 1.0

            losses.append(loss)

        losses = torch.stack(losses)
        loss = torch.min(losses)
        '''

        # print(f"unsupverised loss : {loss}, time_used = {time.time() - start_time}")
        # min_index = torch.argmin(losses).detach().cpu().numpy()
        # losses = losses.detach().cpu().numpy()
        # return loss, min_index, losses

   