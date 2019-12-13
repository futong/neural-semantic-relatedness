import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

from . import Constants


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

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs, conc):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs,conc)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        conc[tree.idx] = torch.cat((torch.flatten(inputs[tree.idx]), torch.flatten(tree.state[0])), 0)[None,:]
        # print("idx",tree.idx)
        return tree.state


class BiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=1, dropout=0.1):
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

# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * 1050, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        out = F.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out), dim=1)
        return out



# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze, att_units, att_hops, maxlen, dropout2):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.attention = SelfAttentiveEncoder(cuda, mem_dim, att_units, att_hops, maxlen, dropout2)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes)
        self.bi_gru = BiGRU(450)

    def forward(self, ltree, linputs, rtree, rinputs):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)
        conc_ten_l = [0 for _ in range(len(linputs))]
        conc_ten_r = [0 for _ in range(len(rinputs))]
        lstate, lhidden = self.childsumtreelstm(ltree, linputs, conc_ten_l)
        rstate, rhidden = self.childsumtreelstm(rtree, rinputs, conc_ten_r)

        l_conc_tensor = torch.cat(conc_ten_l, 0)
        r_conc_tensor = torch.cat(conc_ten_r, 0)
        for j in l_conc_tensor:
            if type(j) == int:
                return torch.tensor([[-1.5829, -0.2299]])
        for j in r_conc_tensor:
            if type(j) == int:
                return torch.tensor([[-1.5829, -0.2299]])
        final_out_l = torch.cat((torch.flatten(lstate),self.bi_gru(l_conc_tensor[None, :, :])),0)
        final_out_r = torch.cat((torch.flatten(rstate),self.bi_gru(r_conc_tensor[None, :, :])),0)
        # print(final_out_l.size(),final_out_r.size())
        output = self.similarity(final_out_l[None,:], final_out_r[None,:])
        # print("lstate",lstate.size())
        # output = self.similarity(lstate, rstate)
        # print(output)
        return output

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

        if penalize and inputs:
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


