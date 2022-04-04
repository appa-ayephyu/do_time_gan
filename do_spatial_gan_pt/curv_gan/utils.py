import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from do_spatial_gan_pt.curv_gan.models import LinearClassifier

cache_filename = "./graph_gan_pt/cache/" + app + "/" + dataset + ".pkl"


def random_walk(node_id, graph, walk_len):
    walk = []
    p = node_id
    while len(walk) < walk_len:
        if p not in graph or len(list(graph.neighbors(p))) == 0:
            break
        p = random.choice(list(graph.neighbors(p)))
        walk.append(p)
    return walk


def eval_link_prediction(embs, data, generator, discriminator, device):
    """
    ROC-AUC Score using F-D Decoder as logits.
    """
    generator.eval()
    discriminator.eval()
    emb_in, emb_out = data["test_edges_pos"][0], data["test_edges_pos"][1]
    pos_scores = discriminator(
        embs.index_select(0, torch.tensor(emb_in).to(device)),
        embs.index_select(0, torch.tensor(emb_out).to(device)),
    )
    emb_in, emb_out = data["test_edges_neg"][0], data["test_edges_neg"][1]
    neg_scores = discriminator(
        embs.index_select(0, torch.tensor(emb_in).to(device)),
        embs.index_select(0, torch.tensor(emb_out).to(device)),
    )
    labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
    preds = list(pos_scores.cpu().data.numpy()) + list(neg_scores.cpu().data.numpy())
    roc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    return roc, ap


def eval_node_classification(embs, data, discriminator, args):
    """
    Micro- and Macro- F1 Score using LinearRegression.
    """
    embs = discriminator.manifold.logmap0(embs, discriminator.c)
    embeddings = embs.cpu().data.numpy().tolist()
    X = []
    Y = []
    for idx, key in enumerate(data["labels"]):
        X.append(embeddings[idx] + embeddings[idx])
        Y.append(key)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=123
    )

    lr = LinearClassifier(args)
    lr.train(torch.tensor(X_train), torch.LongTensor(Y_train))
    Y_pred = lr.test(torch.tensor(X_test))

    micro_f1 = f1_score(Y_test, Y_pred, average="micro")
    macro_f1 = f1_score(Y_test, Y_pred, average="macro")
    return micro_f1, macro_f1
