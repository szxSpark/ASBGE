from __future__ import division
import math
import torch

class Dataset(object):
    def __init__(self, dataset, batchSize, cuda, volatile=False, pointer_gen=False, is_coverage=False):
        assert type(dataset) == dict
        srcData = dataset['src']
        if 'tgt' in dataset:
            tgtData = dataset['tgt']
        else:
            tgtData = None

        self.src = srcData
        if tgtData:
            self.tgt = tgtData
            assert (len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src) / batchSize)
        self.volatile = volatile

        self.pointer_gen = pointer_gen
        self.is_coverage = is_coverage
        if self.pointer_gen:
            self.src_extend_vocab = dataset['src_extend_vocab']
            self.tgt_extend_vocab = dataset['tgt_extend_vocab']
            self.src_oovs_list = dataset['src_oovs_list']

    def _batchify(self, data, start_idx, end_idx, align_right=False, include_lengths=False, is_src=True):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        enc_batch_extend_vocab = None
        dec_batch_extend_vocab = None
        extra_zeros = None
        article_oovs = None
        coverage = None

        if self.pointer_gen:
            if is_src:
                article_oovs = self.src_oovs_list[start_idx:end_idx]
                max_art_oovs = max([len(_article_oovs) for _article_oovs in article_oovs])  # 最长的oovs长度
                src_extend_vocab = self.src_extend_vocab[start_idx:end_idx]
                enc_batch_extend_vocab = data[0].new(len(data), max_length).fill_(utils.Constants.PAD)
                for i in range(len(data)):
                    data_length = data[i].size(0)
                    offset = max_length - data_length if align_right else 0
                    enc_batch_extend_vocab[i].narrow(0, offset, data_length).copy_(torch.LongTensor(src_extend_vocab[i]))
                if max_art_oovs > 0:
                    extra_zeros = torch.zeros((self.batchSize, max_art_oovs))
            else:  # tgt
                tgt_extend_vocab = self.tgt_extend_vocab[start_idx:end_idx]
                dec_batch_extend_vocab = data[0].new(len(data), max_length).fill_(utils.Constants.PAD)
                for i in range(len(data)):
                    data_length = data[i].size(0)
                    offset = max_length - data_length if align_right else 0
                    dec_batch_extend_vocab[i].narrow(0, offset, data_length).copy_(
                        torch.LongTensor(tgt_extend_vocab[i]))

        if self.is_coverage:
            coverage = torch.zeros(len(data), max_length)

        out = data[0].new(len(data), max_length).fill_(utils.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        # out, enc_batch_extend_vocab 的 size相同
        if include_lengths and is_src:  # src
            return out, enc_batch_extend_vocab, extra_zeros, article_oovs, coverage, lengths
        if not include_lengths and not is_src:  # tgt
            return out, dec_batch_extend_vocab

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)

        start_idx = index * self.batchSize
        end_idx = (index + 1) * self.batchSize
        srcBatch, enc_batch_extend_vocab, extra_zeros, article_oovs, coverage, lengths = self._batchify(
            self.src[start_idx:end_idx],
            start_idx=index * self.batchSize,
            end_idx=(index + 1) * self.batchSize,
            align_right=False, include_lengths=True, is_src=True)
        if self.tgt:
            tgtBatch, dec_batch_extend_vocab = self._batchify(
                self.tgt[start_idx:end_idx],
                start_idx=index * self.batchSize,
                end_idx=(index + 1) * self.batchSize,
                is_src=False
            )
        else:
            tgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        if tgtBatch is None:
            if self.pointer_gen:
                if extra_zeros is not None:
                    batch = zip(indices, srcBatch, enc_batch_extend_vocab, extra_zeros, article_oovs, coverage)
                else:
                    batch = zip(indices, srcBatch, enc_batch_extend_vocab, article_oovs, coverage, )
            else:
                batch = zip(indices, srcBatch, )
        else:
            if self.pointer_gen:
                if extra_zeros is not None:
                    batch = zip(indices, srcBatch, tgtBatch, enc_batch_extend_vocab, extra_zeros, article_oovs, coverage, dec_batch_extend_vocab)
                else:
                    batch = zip(indices, srcBatch, tgtBatch, enc_batch_extend_vocab, article_oovs, coverage, dec_batch_extend_vocab)
            else:
                batch = zip(indices, srcBatch, tgtBatch,)

        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            if self.pointer_gen:
                if extra_zeros is not None:
                    indices, srcBatch, enc_batch_extend_vocab, extra_zeros, article_oovs, coverage, = zip(*batch)
                else:
                    indices, srcBatch, enc_batch_extend_vocab, article_oovs, coverage, = zip(*batch)
                    extra_zeros = None
            else:
                indices, srcBatch,  = zip(*batch)
        else:
            if self.pointer_gen:
                if extra_zeros is not None:
                    indices, srcBatch, tgtBatch, enc_batch_extend_vocab, extra_zeros, article_oovs, coverage, dec_batch_extend_vocab = zip(*batch)
                else:
                    indices, srcBatch, tgtBatch, enc_batch_extend_vocab, article_oovs, coverage, dec_batch_extend_vocab = zip(*batch)
                    extra_zeros = None
            else:
                indices, srcBatch, tgtBatch, = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            return b

        lengths = torch.LongTensor(lengths).view(1, -1)
        return (wrap(srcBatch), lengths, wrap(enc_batch_extend_vocab), wrap(extra_zeros), article_oovs, wrap(coverage)), \
               (wrap(tgtBatch), wrap(dec_batch_extend_vocab)), \
               indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        if self.pointer_gen:
            data = list(zip(self.src, self.tgt, self.src_extend_vocab, self.tgt_extend_vocab, self.src_oovs_list))
            self.src, self.tgt, self.src_extend_vocab, self.tgt_extend_vocab, self.src_oovs_list = zip(*[data[i] for i in torch.randperm(len(data))])
        else:
            data = list(zip(self.src, self.tgt))
            self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])
