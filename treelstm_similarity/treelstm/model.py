import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import treelstm.Constants as Constants


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.H = []

    def node_forward(self, inputs, child_c, child_h):
        inputs = torch.unsqueeze(inputs, 0)
        # child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        child_h_sum = torch.sum(child_h, dim=0)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        # c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        c = torch.mul(i, u) + torch.sum(fc, dim=0)
        h = torch.mul(o, F.tanh(c))
        self.H.append(h)
        return c, h

    def forward(self, tree, inputs, conc):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs,conc)

        if tree.num_children == 0:
            child_c = Var(inputs[0].detach().new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].detach().new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        conc[tree.idx] = torch.cat((torch.flatten(inputs[tree.idx]), torch.flatten(tree.state[0])), 0)[None,:]
        # print("idx",tree.idx)
        return tree.state

class BiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=1, dropout=0):
        super(BiGRU, self).__init__()

        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(input_seqs, hidden)
        # outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        final_out = torch.cat((torch.flatten(hidden[0,-1,:]),torch.flatten(hidden[1,-1,:])))

        return final_out


class SelfAttentiveEncoder(nn.Module):
    def __init__(self, cuda, mem_dim, att_units, att_hops, maxlen, dropout2):
        super(SelfAttentiveEncoder, self).__init__()

        self.att_units = att_units
        self.hops = att_hops
        self.len = maxlen
        self.attention_att_hops = att_hops
        self.cudaFlag = cuda
        self.pad_idx = 0
        self.drop = nn.Dropout(dropout2)
        self.ws1 =  nn.Linear(mem_dim, att_units, bias=False)
        self.ws2 =  nn.Linear(att_units, att_hops, bias=False)

        self.init_weights()

        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, outp, inputs=None,penalize=True):

        outp = torch.unsqueeze(outp, 0) # Expects input of the form [btch, len, nhid]

        compressed_embeddings = outp.view(outp.size(1), -1)  # [btch*len, nhid]
        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(1, outp.size(1), -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]

        if penalize and [inputs]:
            # print(outp.size(),inputs.size())
            top = Var(torch.zeros(inputs.size(0), self.hops))
            bottom = Var(torch.ones(outp.size(1) - inputs.size(0), self.hops))

            total = torch.cat((top, bottom), 0)
            total = torch.unsqueeze(torch.transpose(total, 0, 1), 0)
            penalized_term = torch.unsqueeze(total, 0)
            if self.cudaFlag:
                penalized_term = penalized_term.cuda()
            penalized_alphas = torch.add(alphas, -10000 * penalized_term)
        else:
            assert penalize == False and inputs == None
            penalized_alphas = alphas

        # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, outp.size(1)))  # [bsz*hop, len]
        alphas = alphas.view(outp.size(0), self.hops, outp.size(1))  # [hop, len]
        M = torch.bmm(alphas, outp)  # [bsz, hop, mem_dim]
        return M, alphas



class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim1, hidden_dim2, hidden3, num_classes, att_hops, dropout3):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.att_hops = att_hops
        self.drop = nn.Dropout(dropout3)

        self.lat_att = nn.Linear(self.att_hops, 1, bias=True)
        # self.fc = nn.Linear(3*hidden_dim1, hidden_dim2)
        self.fc = nn.Linear(6900, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, hidden3)
        self.out1 = nn.Linear(hidden3, num_classes)

    def forward(self, lvec, rvec,l_gru=Var(torch.zeros((1300),requires_grad=True)),
                r_gru=Var(torch.zeros((1300),requires_grad=True))):
        lvec = self.lat_att(lvec.t()).t()
        rvec = self.lat_att(rvec.t()).t()

        lvec = torch.cat([torch.flatten(lvec),torch.flatten(l_gru)],0)[None,:]
        rvec = torch.cat([torch.flatten(rvec), torch.flatten(r_gru)], 0)[None, :]

        # print("lvec", lvec.size())

        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        mean_dist = 0.5*(torch.add(lvec, rvec))

        fr = torch.cat((abs_dist, mult_dist, mean_dist), 1)

        fc = F.leaky_relu(self.fc(self.drop(fr)))
        fc2 = F.sigmoid(self.out(self.drop(fc)))
        out = F.log_softmax(self.out1(self.drop(fc2)))

        return out



def pad(H, pad, maxlen):
    if H.size(0) > maxlen:
        # print("maxlen",maxlen)
        # print(H)
        return H[0:maxlen]
    elif H.size(0) < maxlen:
        pad = torch.cat([pad] * (maxlen - H.size(0)), 0)
        return torch.cat((H, pad), 0)
    else:
        return H



class SimilarityTreeLSTM(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, hidden_dim1, hidden_dim2,hidden3, num_classes,
                att_hops, att_units, maxlen, dropout1, dropout2, dropout3):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=False)
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.maxlen = maxlen
        self.attention = SelfAttentiveEncoder(cuda, mem_dim, att_units, att_hops, maxlen, dropout2)
        self.similarity = Similarity(mem_dim, hidden_dim1, hidden_dim2, hidden3, num_classes, att_hops, dropout3)

        self.bi_gru = BiGRU(650)
        self.pad_hidden = nn.Parameter(torch.zeros(1, mem_dim))
        self.wf = nn.Parameter(torch.zeros(1, mem_dim, hidden_dim1).uniform_(-1, 1))


    def forward(self, ltree, linputs, rtree, rinputs):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)

        self.childsumtreelstm.H = []
        conc_ten_l = [0 for _ in range(len(linputs))]
        conc_ten_r = [0 for _ in range(len(rinputs))]
        lstate, lhidden = self.childsumtreelstm(ltree, linputs, conc_ten_l)

        Hl = torch.cat(self.childsumtreelstm.H, 0)
        Hl = pad(Hl, lhidden.view(1, -1), self.maxlen)

        self.childsumtreelstm.H = []

        rstate, rhidden = self.childsumtreelstm(rtree, rinputs, conc_ten_r)

        Hr = torch.cat(self.childsumtreelstm.H, 0)
        Hr = pad(Hr, self.pad_hidden, self.maxlen) # [btch, len, mem_dim]

        Ml, attl = self.attention.forward(Hl, linputs)
        Mr, attr = self.attention.forward(Hr, rinputs)  # [btc, hops, mem_dim]

        Ml = F.relu(torch.bmm(Ml, self.wf))
        Mr = F.relu(torch.bmm(Mr, self.wf))

        lstate = torch.squeeze(Ml)
        rstate = torch.squeeze(Mr)

        for j in conc_ten_l:
            if type(j) == int:
                output = self.similarity(lstate, rstate)
                return output
        for j in conc_ten_r:
            if type(j) == int:
                output = self.similarity(lstate, rstate)
                return output

        l_conc_tensor = torch.cat(conc_ten_l, 0)
        r_conc_tensor = torch.cat(conc_ten_r, 0)
        l_gru = self.bi_gru(l_conc_tensor[None, :, :])
        r_gru = self.bi_gru(r_conc_tensor[None, :, :])

        output = self.similarity(lstate, rstate, l_gru, r_gru)
        return output