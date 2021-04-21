from base import *

from odelstm_data import Walker2dImitationData

seqlen = 64
data = Walker2dImitationData(seq_len=seqlen)


class tempf(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels, in_channels)
        # self.dense2 = nn.Linear(in_channels, out_channels)
        # self.dense3 = nn.Linear(out_channels, out_channels)

    def forward(self, h, x):
        out = self.dense1(x)
        # out = self.actv(out)
        # out = self.dense2(out)
        # out = self.actv(out)
        # out = self.dense3(out)
        return out


class temprnn(nn.Module):
    def __init__(self, in_channels, out_channels, nhidden, res=False, cont=False):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels, nhidden)
        self.dense2 = nn.Linear(nhidden, nhidden)
        self.dense2y = nn.Linear(nhidden, nhidden)
        self.dense3 = nn.Linear(nhidden, out_channels)
        self.cont = cont
        self.res = res

    def forward(self, h, x):
        p, q = torch.split(h.clone(), 1, dim=1)
        y = self.dense1(x)
        y = self.actv(y)
        m_ = self.dense2(p) + self.dense2y(y.view(p.shape))
        m_ = self.actv(m_)
        m_ = self.dense3(m_) + q
        out = torch.cat([p, m_], dim=1)
        return out


class MODEL(nn.Module):
    def __init__(self, res=False, cont=False):
        super(MODEL, self).__init__()
        nhid = 84
        self.cell = HeavyBallNODE(tempf(nhid, nhid), corr=0, corrf=False)
        # self.cell = HeavyBallNODE(tempf(nhid, nhid))
        self.rnn = temprnn(17, nhid, nhid, res=res, cont=cont)
        self.ode_rnn = ODE_RNN(self.cell, self.rnn, (2, nhid), tol=1e-7)
        self.outlayer = nn.Linear(nhid, 17)

    def forward(self, t, x):
        out = self.ode_rnn(t, x)
        out = self.outlayer(out[:, :, 0])[1:]
        return out


lr_dict = {0: 0.001, 20: 0.005}
res = True
cont = True
torch.manual_seed(0)
model = MODEL(res=res, cont=cont)
fname = 'output/walker/log_0.txt'
outfile = open(fname, 'w')
outfile.write('res: {}, cont: {}\n'.format(res, cont))
outfile.write(model.__str__())
outfile.write('\n' * 3)
outfile.close()
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[0])
print('Number of Parameters: {}'.format(count_parameters(model)))
timelist = [time.time()]
for epoch in range(1000):
    if epoch in lr_dict:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[epoch])
    model.cell.nfe = 0
    batchsize = 256
    losslist = []
    for b_n in range(0, data.train_x.shape[1], batchsize):
        predict = model(data.train_times[:, b_n:b_n + batchsize] / 64.0, data.train_x[:, b_n:b_n + batchsize])
        loss = criteria(predict, data.train_y[:, b_n:b_n + batchsize])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        losslist.append(loss.detach().cpu().numpy())
        optimizer.step()
    tloss = np.mean(losslist)
    timelist.append(time.time())
    nfe = model.cell.nfe / len(range(0, data.train_x.shape[1], batchsize))
    if epoch == 0 or (epoch + 1) % 1 == 0:
        predict = model(data.valid_times / 64.0, data.valid_x)
        vloss = criteria(predict, data.valid_y)
        vloss = vloss.detach().cpu().numpy()
        outfile = open(fname, 'a')
        outstr = str_rec(['epoch', 'tloss', 'nfe', 'vloss', 'time'],
                         [epoch, tloss, nfe, vloss, timelist[-1] - timelist[-2]])
        print(outstr)
        outfile.write(outstr + '\n')
        outfile.close()
    if epoch == 0 or (epoch + 1) % 20 == 0:
        model.cell.nfe = 0
        predict = model(data.test_times / 64.0, data.test_x)
        sloss = criteria(predict, data.test_y)
        sloss = sloss.detach().cpu().numpy()
        outfile = open(fname, 'a')
        outstr = str_rec(['epoch', 'nfe', 'sloss', 'time'],
                         [epoch, model.cell.nfe, sloss, timelist[-1] - timelist[-2]])
        print(outstr)
        outfile.write(outstr + '\n')
        outfile.close()
        torch.save(model, 'output/walker_hbnode_rnn.mdl')