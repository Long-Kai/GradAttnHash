import torch
import numpy as np


def run_test(query_loader, db_loader, hash_model, autocuda, test_10crop=False, num_query=None, num_top_return=10):
    """
    num_top_return: top k return to calculate precision, recall
    """
    # hash data
    query_set = _hash_data(data_loader=query_loader, hash_model=hash_model, autocuda=autocuda,
                           test_10crop=test_10crop, max_load=num_query)
    db_set = _hash_data(data_loader=db_loader, hash_model=hash_model, test_10crop=test_10crop, autocuda=autocuda)

    # measure performance
    return mean_average_precision(db_code=db_set["code"], db_labels=db_set["label"], query_code=query_set["code"],
                                  query_labels=query_set["label"], R=num_top_return)


def _hash_data(data_loader, hash_model, autocuda, test_10crop=False, max_load=None):
    """hash all the data from data_loader, and return a dict {label,hash}"""
    # hash_ls = []
    # label_ls = []
    if test_10crop:
        iter_test = [iter(data_loader["val"+str(i)]) for i in range(10)]
        loader_len = len(data_loader["val"+str(0)])
        for i in range(loader_len):
            data = [iter_test[j].next() for j in range(10)]
            images = [data[j][0] for j in range(10)]
            labels = data[0][1]
            hash_ls_batch = []
            with torch.no_grad():
                for j in range(10):
                    hash_code = binarized(hash_model(autocuda(images[j])))
                    hash_ls_batch.append(hash_code)
                hash_ls_batch.append(hash_code)  # append one more center crop code
            # hash_ls_batch = sum(hash_ls_batch)
            hash_ls_batch = torch.sign(sum(hash_ls_batch))
            label_ls_batch = labels
            if i == 0:
                batch_size = hash_ls_batch.size(0)
                hash_ls = torch.zeros(loader_len * batch_size, hash_ls_batch.size(1))
                label_ls = torch.zeros(loader_len * batch_size, label_ls_batch.size(1))

            hash_ls[i * batch_size:i * batch_size + hash_ls_batch.size(0)] = hash_ls_batch
            label_ls[i * batch_size:i * batch_size + label_ls_batch.size(0)] = label_ls_batch
    else:
        for i, (images, labels) in enumerate(data_loader):
            with torch.no_grad():
                hash_ls_batch = binarized(hash_model(autocuda(images)))
            label_ls_batch = labels
            if i == 0:
                batch_size = hash_ls_batch.size(0)
                hash_ls = torch.zeros(len(data_loader) * batch_size, hash_ls_batch.size(1))
                label_ls = torch.zeros(len(data_loader) * batch_size, label_ls_batch.size(1))

            hash_ls[i*batch_size:i*batch_size+hash_ls_batch.size(0)] = hash_ls_batch
            label_ls[i*batch_size:i*batch_size+label_ls_batch.size(0)] = label_ls_batch

    if hash_ls_batch.size(0) < batch_size:
        # remove empty row
        max_idx = i * batch_size + hash_ls_batch.size(0)
        hash_ls = hash_ls[0:max_idx]
        label_ls = label_ls[0:max_idx]

    if max_load is not None:
        hash_ls = hash_ls[0:max_load]
        label_ls = label_ls[0:max_load]

    return {"label": label_ls.numpy(), "code": hash_ls.numpy()}


def binarized(feats):
    code = torch.sign(feats).cpu()
    return code


def mean_average_precision(db_code, db_labels, query_code, query_labels, R=1000):

    query_num = query_code.shape[0]

    sim = np.dot(db_code, query_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []

    for i in range(query_num):
        label = query_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(db_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)

    return np.mean(np.array(APx))





