import copy

import torch
import tqdm
from torch.optim import Adam
from time import time

import opt
from utils import *
from encoder import *
from scCAF import scCAF
from data_loader import load_data


def pretrain_ae(model, x):
    print("Pretraining AE...")

    pass

def pretrain_gae(model, x, adj):
    print("Pretraining GAE...")
    pass

def pre_train(model, X1, A1, X2, A2):
    pass
    

def val(model, X1, A1, X2, A2, y):
    model=torch.load('./model_trained/model.pth', map_location=opt.args.device)
    with torch.no_grad():
        X_hat1, Z_hat1, A_hat1, X_hat2, Z_hat2, A_hat2, Q1, Q2, Z1, Z2, cons = model(X1, A1, X2, A2)
        ari, nmi, ami, acc, y_pred = assignment((Q1[0] + Q2[0]).data, y)
        print("ARI: {:.4f}, NMI: {:.4f}, AMI: {:.4f}, ACC: {:.4f}".format(ari, nmi, ami, acc))
        np.save('./output/{}/seed{}_label.npy'.format(opt.args.name, opt.args.seed), y_pred)
        np.save('./output/{}/seed{}_z.npy'.format(opt.args.name, opt.args.seed), ((Z1 + Z2) / 2).cpu().detach().numpy())

if __name__ == '__main__':
    # setup


        print("setting:")

        setup_seed(opt.args.seed)
        opt.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # load data
        Xr, y, Ar = load_data(opt.args.name, 'RNA', opt.args.method, opt.args.k, show_details=False)
        Xa, y, Aa = load_data(opt.args.name, 'ATAC', opt.args.method, opt.args.k, show_details=False)
        opt.args.n_clusters = int(max(y) - min(y) + 1)

        Xr = numpy_to_torch(Xr).to(opt.args.device)
        Ar = numpy_to_torch(Ar, sparse=True).to(opt.args.device)

        Xa = numpy_to_torch(Xa).to(opt.args.device)
        Aa = numpy_to_torch(Aa, sparse=True).to(opt.args.device)

        ae1 = AE(
            ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2,
            ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2,
            n_input=opt.args.n_d1, n_z=opt.args.n_z).to(opt.args.device)

        ae2 = AE(
            ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2,
            ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2,
            n_input=opt.args.n_d2, n_z=opt.args.n_z).to(opt.args.device)

        if opt.args.pretrain:
            opt.args.dropout = 0.4
        gae1 = IGAE(
            gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
            gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
            n_input=opt.args.n_d1, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)

        gae2 = IGAE(
            gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
            gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
            n_input=opt.args.n_d2, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)

        setup_seed(opt.args.seed)
        t0 = time()
        model = scCAF(ae1, ae2, gae1, gae2, n_node=Xr.shape[0]).to(opt.args.device)
        val(model, Xr, Ar, Xa, Aa, y)
        t1 = time()
        print("Time_cost: {}".format(t1 - t0))

