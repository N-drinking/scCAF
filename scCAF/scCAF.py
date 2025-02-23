import opt
from encoder import *
class GatedDeep(nn.Module):
    def __init__(self,num_gate_layes,input_size):
        super(GatedDeep,self).__init__()
        self.num_gate_layes=num_gate_layes
        self.non_linear=nn.ModuleList(([nn.Linear(input_size,input_size).to('cuda:0') for _ in range((self.num_gate_layes))]))
        self.linear = nn.ModuleList(
            ([nn.Linear(input_size, input_size).to('cuda:0') for _ in range((self.num_gate_layes))]))
        self.gate = nn.ModuleList(
            ([nn.Linear(input_size, input_size).to('cuda:0') for _ in range((self.num_gate_layes))]))
        self.act = nn.ReLU()
    def forward(self,z):
        for i in range(self.num_gate_layes):
            gate=torch.sigmoid(self.gate[i](z))
            non_linear=self.act(self.non_linear[i](z))
            linear=self.linear[i](z)
            z_gate=gate*non_linear+(1-gate)*linear
        output=z+z_gate
        return output
class scCAF(nn.Module):
    def __init__(self, ae1, ae2, gae1, gae2, n_node=None):
        super(scCAF, self).__init__()

        self.ae1 = ae1
        self.ae2 = ae2

        self.gae1 = gae1
        self.gae2 = gae2
        self.GatedDeep=GatedDeep(opt.args.h,20)
        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), 0.5),
                           requires_grad=True)  # Z_ae, Z_igae
        self.alpha = Parameter(torch.zeros(1))  # ZG, ZL
        self.bate = Parameter(torch.zeros(1))
        self.cluster_centers1 = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)
        self.cluster_centers2 = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_centers1.data)
        torch.nn.init.xavier_normal_(self.cluster_centers2.data)
        self.q_distribution1 = q_distribution(self.cluster_centers1)
        self.q_distribution2 = q_distribution(self.cluster_centers2)

        self.label_contrastive_module = nn.Sequential(
            nn.Linear(n_node, opt.args.n_clusters),
            nn.Softmax(dim=1)
        )

    def emb_fusion(self, adj1, z_ae1, z_igae1, adj2, z_ae2, z_igae2):
        z_i_1 = self.a * z_ae1 + (1 - self.a) * z_igae1
        z_l_1 = torch.spmm(adj1, z_i_1)
        s1 = torch.mm(z_l_1, z_l_1.t())
        s1 = F.softmax(s1, dim=1)
        z_i_2 = self.a * z_ae2 + (1 - self.a) * z_igae2
        z_l_2 = torch.spmm(adj2, z_i_2)
        s2 = torch.mm(z_l_2, z_l_2.t())
        s2 = F.softmax(s2, dim=1)
        z_g_s = torch.mm(s1, z_l_1)
        z_g_c = torch.mm(s2, z_l_1)
        z_g_c=self.GatedDeep(z_g_c)
        z_g_s = self.GatedDeep(z_g_s)
        # z_tilde = self.alpha * z_g_s + z_l_1
        z_tilde = self.alpha * z_g_s + self.bate * z_g_c + z_l_1
        return z_tilde

    def forward(self, x1, adj1, x2, adj2, pretrain=False):
        # node embedding encoded by AE
        z_ae1 = self.ae1.encoder(x1)
        z_ae2 = self.ae2.encoder(x2)

        # node embedding encoded by IGAE
        z_igae1, a_igae1 = self.gae1.encoder(x1, adj1)
        z_igae2, a_igae2 = self.gae2.encoder(x2, adj2)

        z1 = self.emb_fusion(adj1, z_ae1, z_igae1, adj2, z_ae2, z_igae2)
        z2 = self.emb_fusion(adj2, z_ae2, z_igae2, adj1, z_ae1, z_igae1)

        z1_tilde = self.label_contrastive_module(z1.T)
        z2_tilde = self.label_contrastive_module(z2.T)

        cons = [z1, z2, z1_tilde, z2_tilde]

        # AE decoding
        x_hat1 = self.ae1.decoder(z1)
        x_hat2 = self.ae2.decoder(z2)

        # IGAE decoding
        z_hat1, z_adj_hat1 = self.gae1.decoder(z1, adj1)
        a_hat1 = a_igae1 + z_adj_hat1

        z_hat2, z_adj_hat2 = self.gae2.decoder(z2, adj2)
        a_hat2 = a_igae2 + z_adj_hat2

        if not pretrain:
            # the soft assignment distribution Q
            Q1 = self.q_distribution1(z1, z_ae1, z_igae1)
            Q2 = self.q_distribution2(z2, z_ae2, z_igae2)
        else:
            Q1, Q2 = None, None

        return x_hat1, z_hat1, a_hat1, x_hat2, z_hat2, a_hat2, Q1, Q2, z1, z2, cons
