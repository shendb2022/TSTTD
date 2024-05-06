from typing import Dict
import torch
from Data import Data
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from Scheduler import GradualWarmupScheduler
from Model import SpectralGroupAttention
import os
from Tools import checkFile, standard
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from ts_generation import ts_generation
from sklearn import metrics
import random

def seed_torch(seed=1):
    '''
    Keep the seed fixed thus the results can keep stable
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def spectral_group(x, n, m):
    ### divide the spectrum into n overlapping groups
    pad_size = m // 2
    new_sample = np.pad(x, ((0, 0), (pad_size, pad_size)),
                        mode='symmetric')
    b = x.shape[0]
    group_spectra = np.zeros([b, n, m])
    for i in range(n):
        group_spectra[:, i, :] = np.squeeze(new_sample[:, i:i + m])

    return torch.from_numpy(group_spectra).float()


def cosin_similarity(x, y):
    assert x.shape[1] == y.shape[1]
    x_norm = torch.sqrt(torch.sum(x ** 2, dim=1))
    y_norm = torch.sqrt(torch.sum(y ** 2, dim=1))
    x_y_dot = torch.sum(torch.multiply(x, y), dim=1)
    return x_y_dot / (x_norm * y_norm + 1e-8)


def cosin_similarity_numpy(x, y):
    assert x.shape[1] == y.shape[1]
    x_norm = np.sqrt(np.sum(x ** 2, axis=1))
    y_norm = np.sqrt(np.sum(y ** 2, axis=1))
    x_y = np.sum(np.multiply(x, y), axis=1)
    return x_y / (x_norm * y_norm + 1e-8)


def isia_loss(x, batch_size, margin=1.0, lambd=1):
    '''
    This function is used to calculate the intercategory separation and intracategory aggregation loss
    It includes the triplet loss and cross-entropy loss
    '''
    positive, negative, prior = x[:batch_size], x[batch_size:2 * batch_size], x[2 * batch_size:]
    p_sim = cosin_similarity(positive, prior)
    n_sim1 = cosin_similarity(negative, prior)
    n_sim2 = cosin_similarity(negative, positive)
    max_n_sim = torch.maximum(n_sim1, n_sim2)

    ## triplet loss to maximize the feature distance between anchor and positive samples
    ## while minimizing the feature distance between anchor and negative samples
    triplet_loss = margin + max_n_sim - p_sim
    triplet_loss = torch.relu(triplet_loss)
    triplet_loss = torch.mean(triplet_loss)

    ## binary cross-entropy loss to distinguish pixels of background and target
    p_sim = torch.sigmoid(p_sim)
    n_sim = torch.sigmoid(1 - n_sim1)
    bce_loss = -0.5 * torch.mean(torch.log(p_sim + 1e-8) + torch.log(n_sim + 1e-8))

    isia_loss = triplet_loss + lambd * bce_loss

    return isia_loss


def paintTrend(losslist, epochs=100, stride=10):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.title('loss-trend')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(np.arange(0, epochs, stride))
    plt.xlim(0, epochs)
    plt.plot(losslist, color='r')
    plt.show()


def train(modelConfig: Dict):
    seed_torch(modelConfig['seed'])
    device = torch.device(modelConfig["device"])
    dataset = Data(modelConfig["path"])
    dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True,
                            pin_memory=True)
    # model setup
    net_model = SpectralGroupAttention(band=modelConfig['band'], m=modelConfig['group_length'],
                                       d=modelConfig['channel'], depth=modelConfig['depth'],heads=modelConfig['heads'],
                                       dim_head=modelConfig['dim_head'], mlp_dim=modelConfig['mlp_dim'], adjust=modelConfig['adjust']).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    path = modelConfig["save_dir"] + '/' + modelConfig['path'] + '/'
    checkFile(path)

    # start training
    net_model.train()
    loss_list = []
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for positive, negative in tqdmDataLoader:
                # train
                combined_vectors = np.concatenate([positive, negative, dataset.target_spectrum], axis=0)
                combined_groups = spectral_group(combined_vectors, modelConfig['band'], modelConfig['group_length'])
                optimizer.zero_grad()
                x_0 = combined_groups.to(device)
                features = net_model(x_0)
                loss = isia_loss(features, modelConfig['batch_size'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            path, 'ckpt_' + str(e) + "_.pt"))
        loss_list.append(loss.item())
    paintTrend(loss_list, epochs=modelConfig['epoch'], stride=5)

def select_best(modelConfig: Dict):
    seed_torch(modelConfig['seed'])
    device = torch.device(modelConfig["device"])
    opt_epoch = 0
    max_auc = 0
    path = modelConfig["save_dir"] + '/' + modelConfig['path'] + '/'
    for e in range(modelConfig['epoch']):
        with torch.no_grad():
            mat = sio.loadmat(modelConfig["path"])
            data = mat['data']
            map = mat['map']
            data = standard(data)
            data = np.float32(data)
            target_spectrum = ts_generation(data, map, 7)
            h, w, c = data.shape
            numpixel = h * w
            data_matrix = np.reshape(data, [-1, c], order='F')
            model = SpectralGroupAttention(band=modelConfig['band'], m=modelConfig['group_length'],
                                           d=modelConfig['channel'], depth=modelConfig['depth'],
                                           heads=modelConfig['heads'],
                                           dim_head=modelConfig['dim_head'], mlp_dim=modelConfig['mlp_dim'],
                                           adjust=modelConfig['adjust']).to(device)
            ckpt = torch.load(os.path.join(
                path, "ckpt_%s_.pt" % e), map_location=device)
            model.load_state_dict(ckpt)
            print("model load weight done.%s" % e)
            model.eval()

            batch_size = modelConfig['batch_size']
            detection_map = np.zeros([numpixel])
            target_prior = spectral_group(target_spectrum.T, modelConfig['band'], modelConfig['group_length'])
            target_prior = target_prior.to(device)
            target_features = model(target_prior)
            target_features = target_features.cpu().detach().numpy()

            for i in range(0, numpixel - batch_size, batch_size):
                pixels = data_matrix[i:i + batch_size]
                pixels = spectral_group(pixels, modelConfig['band'], modelConfig['group_length'])
                pixels = pixels.to(device)
                features = model(pixels)
                features = features.cpu().detach().numpy()
                detection_map[i:i + batch_size] = cosin_similarity_numpy(features, target_features)

            left_num = numpixel % batch_size
            if left_num != 0:
                pixels = data_matrix[-left_num:]
                pixels = spectral_group(pixels, modelConfig['band'], modelConfig['group_length'])
                pixels = pixels.to(device)
                features = model(pixels)
                features = features.cpu().detach().numpy()
                detection_map[-left_num:] = cosin_similarity_numpy(features, target_features)

            detection_map = np.reshape(detection_map, [h, w], order='F')
            detection_map = standard(detection_map)
            detection_map = np.clip(detection_map, 0, 1)
            y_l = np.reshape(map, [-1, 1], order='F')
            y_p = np.reshape(detection_map, [-1, 1], order='F')

            ## calculate the AUC value
            fpr, tpr, _ = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
            fpr = fpr[1:]
            tpr = tpr[1:]
            auc = round(metrics.auc(fpr, tpr), modelConfig['epision'])
            if auc > max_auc:
                max_auc = auc
                opt_epoch = e
    print(max_auc)
    print(opt_epoch)

def eval(modelConfig: Dict):
    seed_torch(modelConfig['seed'])
    device = torch.device(modelConfig["device"])
    path = modelConfig["save_dir"] + '/' + modelConfig['path'] + '/'
    with torch.no_grad():
        mat = sio.loadmat(modelConfig["path"])
        data = mat['data']
        map = mat['map']
        data = standard(data)
        data = np.float32(data)
        target_spectrum = ts_generation(data, map, 7)
        h, w, c = data.shape
        numpixel = h * w
        data_matrix = np.reshape(data, [-1, c], order='F')
        model = SpectralGroupAttention(band=modelConfig['band'], m=modelConfig['group_length'],
                                       d=modelConfig['channel'], depth=modelConfig['depth'],heads=modelConfig['heads'],
                                       dim_head=modelConfig['dim_head'], mlp_dim=modelConfig['mlp_dim'], adjust=modelConfig['adjust']).to(device)
        ckpt = torch.load(os.path.join(
            path, modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()

        batch_size = modelConfig['batch_size']
        detection_map = np.zeros([numpixel])
        target_prior = spectral_group(target_spectrum.T, modelConfig['band'], modelConfig['group_length'])
        target_prior = target_prior.to(device)
        target_features = model(target_prior)
        target_features = target_features.cpu().detach().numpy()

        for i in range(0, numpixel - batch_size, batch_size):
            pixels = data_matrix[i:i + batch_size]
            pixels = spectral_group(pixels, modelConfig['band'], modelConfig['group_length'])
            pixels = pixels.to(device)
            features = model(pixels)
            features = features.cpu().detach().numpy()
            detection_map[i:i + batch_size] = cosin_similarity_numpy(features, target_features)

        left_num = numpixel % batch_size
        if left_num != 0:
            pixels = data_matrix[-left_num:]
            pixels = spectral_group(pixels, modelConfig['band'], modelConfig['group_length'])
            pixels = pixels.to(device)
            features = model(pixels)
            features = features.cpu().detach().numpy()
            detection_map[-left_num:] = cosin_similarity_numpy(features, target_features)

        detection_map = np.reshape(detection_map, [h, w], order='F')
        detection_map = standard(detection_map)
        detection_map = np.clip(detection_map, 0, 1)
        # plt.imshow(detection_map)
        # plt.show()
        y_l = np.reshape(map, [-1, 1], order='F')
        y_p = np.reshape(detection_map, [-1, 1], order='F')

        ## calculate the AUC value
        fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
        fpr = fpr[1:]
        tpr = tpr[1:]
        threshold = threshold[1:]
        auc1 = round(metrics.auc(fpr, tpr), modelConfig['epision'])
        auc2 = round(metrics.auc(threshold, fpr), modelConfig['epision'])
        auc3 = round(metrics.auc(threshold, tpr), modelConfig['epision'])
        auc4 = round(auc1 + auc3 - auc2, modelConfig['epision'])
        auc5 = round(auc3 / auc2, modelConfig['epision'])
        print('{:.{precision}f}'.format(auc1, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc2, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc3, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc4, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc5, precision=modelConfig['epision']))

        plt.imshow(detection_map)
        plt.show()
