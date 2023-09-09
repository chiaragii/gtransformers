"""
    IMPORTING LIBS
"""

import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

import matplotlib.pyplot as plt

from tabulate import tabulate


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.BPI_graph_classification.load_net import gnn_model
from data.data import LoadData

"""
    GPU Setup
"""


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""


def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""


def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    t0 = time.time()
    per_epoch_time = []

    DATASET_NAME = dataset.name

    # label proportion in train
    label_proportions_train, actual_labels_train, train_samples, deleted_labels_train = \
        dataset.check_class_imbalance(dataset.train.deleted_labels, dataset.train.label_dict,
                                      dataset.train.graph_labels, net_params['n_classes'])


    # label proportion in test
    label_proportions_test, actual_labels_test, test_samples, deleted_labels_test = \
        dataset.check_class_imbalance(dataset.test.deleted_labels, dataset.test.label_dict,
                                      dataset.test.graph_labels, net_params['n_classes'])

    # label proportion in val
    label_proportions_val, actual_labels_val, val_samples, deleted_labels_val = \
        dataset.check_class_imbalance(dataset.val.deleted_labels, dataset.val.label_dict,
                                      dataset.val.graph_labels, net_params['n_classes'])

    if net_params['lap_pos_enc']:
        st = time.time()
        print("[!] Adding Laplacian positional encoding.")
        dataset._add_laplacian_positional_encodings(net_params['pos_enc_dim'])
        print('Time LapPE:', time.time() - st)

    if net_params['wl_pos_enc']:
        st = time.time()
        print("[!] Adding WL positional encoding.")
        dataset._add_wl_positional_encodings()
        print('Time WL PE:', time.time() - st)

    if net_params['full_graph']:
        st = time.time()
        print("[!] Converting the given graphs to full graphs..")
        dataset._make_full_graph()
        print('Time taken to convert to full graphs:', time.time() - st)

    trainset, testset, valset = dataset.train, dataset.test, dataset.val

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(
            DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    epoch_train_losses, epoch_val_losses, epoch_test_losses = [], [], []
    epoch_train_accs, epoch_val_accs, epoch_test_accs = [], [], []
    epoch_train_f1s, epoch_val_f1s, epoch_test_f1s = [], [], []
    epoch_train_wf1s, epoch_test_wf1s = [], []
    epoch_count = []
    first_epoch = 0

    # import train and evaluate functions
    from train.train_BPI_graph_classification import train_epoch, evaluate_network

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    # At any point you can hit Ctrl + C to break out of training early.
    with tqdm(range(params['epochs'])) as t:
        for epoch in t:
            try:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_acc, epoch_train_f1, epoch_train_conf, wf1_train, optimizer = train_epoch(model, optimizer, device,
                                                                                           train_loader,
                                                                                           epoch, actual_labels_train, train_samples)

                epoch_val_loss, epoch_val_acc, epoch_val_f1, epoch_val_conf, f1_per_class_val, wf1_val = evaluate_network(model,
                                                                                                                 device, val_loader, epoch,
                                                                                                                 actual_labels_val, val_samples)
                epoch_test_loss, epoch_test_acc, epoch_test_f1, epoch_test_conf, f1_per_class_test, wf1_test = evaluate_network(model, device,
                                                                                                                      test_loader, epoch,
                                                                                                                      actual_labels_test, test_samples)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_test_losses.append(epoch_test_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)
                epoch_test_accs.append(epoch_test_acc)
                epoch_train_f1s.append(epoch_train_f1)
                epoch_val_f1s.append(epoch_val_f1)
                epoch_test_f1s.append(epoch_test_f1)
                epoch_train_wf1s.append(wf1_train)
                epoch_test_wf1s.append(wf1_test)

                epoch_count.append(first_epoch)
                first_epoch += 1

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                writer.add_scalar('train/_f1', epoch_train_f1, epoch)
                writer.add_scalar('val/_f1', epoch_val_f1, epoch)
                writer.add_scalar('test/_f1', epoch_test_f1, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                              test_acc=epoch_test_acc, train_f1=epoch_train_f1,
                              val_f1=epoch_val_f1, test_f1=epoch_test_f1)

                per_epoch_time.append(time.time() - start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - t0 > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break

            except KeyboardInterrupt:
                print('-' * 89)
                print('Exiting from training early because of KeyboardInterrupt')
                break

            except Exception:
                break

    _, test_acc, test_f1, confusion_test, f1s_per_class_test, weighted_f1_test = evaluate_network(model, device, test_loader, epoch, actual_labels_test, test_samples)
    _, train_acc, train_f1, confusion_train, f1s_per_class_train, weighted_f1_train = evaluate_network(model, device, train_loader, epoch, actual_labels_train, train_samples)

    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Train Accuracy: {:.4f}".format(train_acc))
    print("Test F1-score: {:.4f}".format(test_f1))
    print("Train F1-score: {:.4f}".format(train_f1))
    print("Weighted Test F1-score: {:.4f}".format(weighted_f1_test))
    print("Weighted Train F1-score: {:.4f}".format(weighted_f1_train))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:

        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n"""
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param']))

        f.write("""\n\n\nTraining graphs: {}\nTest graphs: {} \nValidation graphs: {}\n""".format(len(trainset),
                                                                                                  len(testset),
                                                                                                  len(valset)))

        f.write("""\n\nFINAL RESULTS\nTEST ACCURACY: {:.4f}%\nTRAIN ACCURACY: {:.4f}%\nTEST F1-SCORE: {:.4f}%\nTRAIN F1-SCORE: {:.4f}%
        \n\nConvergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nNum Epochs: {}\nAverage Time Per Epoch: {:.4f} s\n\n\n"""
                .format(np.mean(np.array(test_acc)), np.mean(np.array(train_acc)), np.mean(np.array(test_f1)),
                        np.mean(np.array(train_f1)), epoch, (time.time() - t0) / 3600, first_epoch,
                        np.mean(per_epoch_time)))

        f.write('\n<------------------------------------- Test Results ------------------------------------->\n\n')
        f.write("""Testset Confusion Matrix:\n""")
        table = tabulate(confusion_test, tablefmt='grid')
        f.write(table + '\n')


        f.write("""\n Test F1-score per class: \n""")
        data = [(item, score) for item, score in zip(actual_labels_test, f1s_per_class_test)]
        table = tabulate(data, headers=["Class", "F1-score"], tablefmt='grid')
        f.write(table)

        f.write("""\n\nWeighted Test F1-scores per class: {:.4f}%\n""".format(weighted_f1_test))

        f.write("""\nClass distribution in testset:\n""")
        data = [(i, test_samples[i]) for i in test_samples]
        table = tabulate(data, headers=["Class", "Class Distribution"], tablefmt='grid')
        f.write(table)

        f.write("""\n\nClass probabilities in testset:\n""")
        data = [(i, label_proportions_test[i]) for i in label_proportions_test]
        table = tabulate(data, headers=["Class", "Class Probabilities"], tablefmt='grid')
        f.write(table)

        f.write("""\n\nLabels deleted in testset:\n""")
        data = [(i, deleted_labels_test[i]) for i in deleted_labels_test]
        table = tabulate(data, headers=["Class", "Deleted Samples"], tablefmt='grid')
        f.write(table)


        f.write('\n\n<------------------------------------- Train Results ------------------------------------->\n\n')

        f.write("""Trainset Confusion Matrix:\n""")
        table = tabulate(confusion_train, tablefmt='grid')
        f.write(table + '\n')

        f.write("""\n Train F1-score per class: \n""")
        data = [(item, score) for item, score in zip(actual_labels_train, f1s_per_class_train)]
        table = tabulate(data, headers=["Class", "F1-score"], tablefmt='grid')
        f.write(table)

        f.write("""\n\nWeighted Train F1-scores per class: {:.4f}%\n""".format(weighted_f1_train))

        f.write("""\nClass distributions in trainset:\n""")
        data = [(i, train_samples[i]) for i in train_samples]
        table = tabulate(data, headers=["Class", "Class Distribution"], tablefmt='grid')
        f.write(table)

        f.write("""\n\nClass probabilities in trainset:\n""")
        data = [(i, label_proportions_train[i]) for i in label_proportions_train]
        table = tabulate(data, headers=["Class", "Class Probabilities"], tablefmt='grid')
        f.write(table)

        f.write("""\n\nLabels deleted in trainset:\n""")
        data = [(i, deleted_labels_train[i]) for i in deleted_labels_train]
        table = tabulate(data, headers=["Class", "Deleted Samples"], tablefmt='grid')
        f.write(table)

        plt.subplot(3, 1, 1)
        plt.plot(epoch_count, epoch_train_accs, label='train acc')
        plt.plot(epoch_count, epoch_test_accs, label='test acc')
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy')

        plt.subplot(3, 1, 2)
        plt.plot(epoch_count, epoch_train_f1s, label='train f1')
        plt.plot(epoch_count, epoch_test_f1s, label='test f1')
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylabel('F1')
        plt.title('Training and Test f1')

        plt.subplot(3, 1, 3)
        plt.plot(epoch_count, epoch_train_wf1s, label='train w-f1')
        plt.plot(epoch_count, epoch_test_wf1s, label='test w-f1')
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylabel('weighted F1')
        plt.title('Training and Test weighted f1')

        plt.tight_layout()
        plt.show()


def main():
    """
        USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--lap_pos_enc', help="Please give a value for lap_pos_enc")
    parser.add_argument('--wl_pos_enc', help="Please give a value for wl_pos_enc")
    args = parser.parse_args()
    with open('configs/GraphsTransformer.json') as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        ['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']

    dataset = LoadData(DATASET_NAME, config['params']['num_nodes'])
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm == 'True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop == 'True' else False
    if args.lap_pos_enc is not None:
        net_params['lap_pos_enc'] = True if args.pos_enc == 'True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)
    if args.wl_pos_enc is not None:
        net_params['wl_pos_enc'] = True if args.pos_enc == 'True' else False

    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)  # node_dim (feat is an integer)
    net_params['n_classes'] = len(dataset.train.label_dict.keys())

    # net_params['num_bond_type'] = dataset.num_bond_type

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)


    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)


main()
