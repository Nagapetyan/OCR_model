# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output
   

class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=71):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.comb = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        #print(embedded.size())
        #print(encoder_outputs.size()) # 26 x 32 x 256
        input_comb = self.comb(torch.cat([embedded, encoder_output], 1))
        hidden = self.gru(input_comb, hidden)
        output = F.log_softmax(self.out(hidden), dim=1)
        #print("out:", output.size()) # 32 x 39

        return output, hidden, None # For compatibility

    def initHidden(self, batch_size):
        result = torch.zeros(batch_size, self.hidden_size)

        return result
    

class CNN(nn.Module):

    def __init__(self, imgH, nc, nh, rcnn_depth=1):
        super(CNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.rcnn_depth = rcnn_depth # depth of recursive CNN

        self.cnn_pool_1 = nn.Sequential(
            nn.Conv2d(nc, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        ) # 64 x 16 x 50
    
        self.cnn1_tied = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True)
        ) # 64 x 16 x 50
        
        self.cnn_pool_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        ) # 128 x 8 x 25
        
        self.cnn2_tied = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        ) # 128 x 8 x 25
        
        self.cnn_pool_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        ) # 256 x 4 x 25
        
        self.cnn3_tied = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True)
        ) # 256 x 4 x 25
        
        self.cnn_pool_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2,2), (2,1), (0,1))
        ) # 256 x 2 x 25
        
        self.cnn4_tied = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True)
        ) # 512 x 2 x 25

        self.cnn_last = nn.Sequential(
            nn.Conv2d(512, 512, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        ) # 512 x 1 x 25

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh)
        )

    def forward(self, input):
        # Block 1
        conv = self.cnn_pool_1(input)

        for _ in range(self.rcnn_depth):
            conv = self.cnn1_tied(conv)
        # Block 2
        conv = self.cnn_pool_2(conv)

        for _ in range(self.rcnn_depth):
            conv = self.cnn2_tied(conv)
        # Block 3
        conv = self.cnn_pool_3(conv)

        for _ in range(self.rcnn_depth):
            conv = self.cnn3_tied(conv)
        # Block 4
        conv = self.cnn_pool_4(conv)

        for _ in range(self.rcnn_depth):
            conv = self.cnn4_tied(conv)
            
        conv = self.cnn_last(conv)

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features calculate
        encoder_outputs = self.rnn(conv)          # seq * batch * n_classes // 25 × batchsize × 256
        
        return encoder_outputs
