"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import numpy as np
from train.metrics import accuracy_TU as accuracy
from train.metrics import accuracy_VOC as f1_score
import math


def train_epoch(model, optimizer, device, data_loader, epoch, actual_labels):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    epoch_train_conf = 0
    epoch_train_f1 = 0
    nb_data = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0; sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None
            
        try:
            batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
        except:
            batch_wl_pos_enc = None

        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
    
        loss = model.loss(batch_scores, torch.flatten(batch_labels))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        acc, conf = accuracy(batch_scores, batch_labels, list(actual_labels.values()))
        epoch_train_acc += acc
        epoch_train_conf += conf
        f1, _ = f1_score(batch_scores, batch_labels, list(actual_labels.values()))
        epoch_train_f1 += f1
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    epoch_train_f1 /= (iter + 1)
    epoch_train_f1 = epoch_train_f1*100
    epoch_train_acc = epoch_train_acc*100

    return epoch_loss, epoch_train_acc, epoch_train_f1, epoch_train_conf, optimizer


def evaluate_network(model, device, data_loader, epoch, actual_labels, class_proportions):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    epoch_test_f1 = 0
    epoch_test_conf = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None
            
            try:
                batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
            except:
                batch_wl_pos_enc = None
                
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
            loss = model.loss(batch_scores, torch.flatten(batch_labels))
            epoch_test_loss += loss.detach().item()
            acc, conf = accuracy(batch_scores, batch_labels, list(actual_labels.values()))
            epoch_test_acc += acc
            epoch_test_conf += conf
            f1, _ = f1_score(batch_scores, batch_labels, list(actual_labels.values()))
            epoch_test_f1 += f1
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        epoch_test_f1 /= (iter + 1)
        epoch_test_f1 = epoch_test_f1*100
        epoch_test_acc = epoch_test_acc*100

        tp = np.diag(epoch_test_conf)
        fp = np.sum(epoch_test_conf, axis=0) - tp
        fn = np.sum(epoch_test_conf, axis=1) - tp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_per_class = 2*precision*recall/(precision+recall)

        acc = 0
        for i in range (0,len(f1_per_class)):
            if math.isnan(f1_per_class[i]):
                acc += 0
            else:
                acc += f1_per_class[i]*list(class_proportions.values())[i]
        weighted_f1 = acc/sum(list(class_proportions.values()))


    return epoch_test_loss, epoch_test_acc, epoch_test_f1, epoch_test_conf, f1_per_class, weighted_f1


