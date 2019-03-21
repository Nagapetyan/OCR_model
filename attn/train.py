# coding:utf-8
import argparse
import random
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
import os
import utils
from dataset import WordImageDataset
from torch.utils.data import DataLoader
import time
import torch.nn as nn
import string
import torchvision.transforms as transforms
import models

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=21, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--encoder', type=str, default='', help="path to encoder (to continue training)")
parser.add_argument('--decoder', type=str, default='', help='path to decoder (to continue training)')
parser.add_argument('--experiment', default='./expr/attn', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--valInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--teaching_forcing_prob', type=float, default=0.5, help='where to use teach forcing')
parser.add_argument('--max_width', type=int, default=26, help='the width of the featuremap out from cnn')
opt = parser.parse_args()

SOS_token = 0
EOS = 1
BLANK = 2 # blank for padding

if opt.experiment is None:
    opt.experiment = 'expr'

os.system('mkdir -p {0}'.format(opt.experiment))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir = '/media/kopoden/5C1E16301E1603A4/Datasets/synth90k'

alphabet = string.printable[:36]
converter = utils.strLabelConverterForAttention(alphabet)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

train_dataset = WordImageDataset(root_dir, "annotation_train_new.txt", alphabet, transform, max_samples=300000)
val_dataset = WordImageDataset(root_dir, "annotation_val_new.txt", alphabet, transform, max_samples=5000)

train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1) # BATCH SIZE = 1!!! Important

nclass = len(alphabet) + 3 # + SOS, EOS, blank
input_channels = 1
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.NLLLoss().to(device)

def weights_init(model):
    # Official init from torch tutorial
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)

encoder = models.CNN(opt.imgH, input_channels, opt.nh)
decoder = models.AttentionDecoder(opt.nh, nclass, dropout_p=0.1, max_length=opt.max_width)

encoder.apply(weights_init)
decoder.apply(weights_init)

encoder = encoder.to(device)
decoder = decoder.to(device)
# continue training or use the pretrained model to initial the parameters of the encoder and decoder
if opt.encoder:
    print('loading pretrained encoder model from %s' % opt.encoder)
    encoder.load_state_dict(torch.load(opt.encoder))
if opt.decoder:
    print('loading pretrained encoder model from %s' % opt.decoder)
    encoder.load_state_dict(torch.load(opt.encoder))

length = torch.IntTensor(opt.batchSize).to(device)

# loss averager
loss_avg = utils.averager()

# setup optimizer
encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def val(encoder, decoder, criterion, val_loader, device):
    print('Start val')

    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        i = 0
        n_correct = 0
        n_total = 0
        loss_avg = utils.averager()

        for data in val_loader:
            i += 1
            cpu_images, cpu_texts = data
            b = cpu_images.size(0)
            image = cpu_images.to(device)
            target_variable = converter.encode(cpu_texts).to(device)
            n_total += len(cpu_texts[0]) + 1
            #print(cpu_images.size(), target_variable.size())

            decoded_words = []
            decoded_label = []
            decoder_attentions = torch.zeros(len(cpu_texts[0]) + 1, opt.max_width)
            #print(decoder_attentions.size())
            encoder_outputs = encoder(image)
            decoder_input = target_variable[0].to(device)
            decoder_hidden = decoder.initHidden(b).to(device)
            #print(encoder_outputs.size(), decoder_input.size(), decoder_hidden.size())
            loss = 0.0

            for di in range(1, target_variable.shape[0]):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                loss_avg.add(loss)
                #print(decoder_attention.data.size())
                decoder_attentions[di-1] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                ni = topi.squeeze(1)
                decoder_input = ni

                if ni == EOS:
                    decoded_words.append('<EOS>')
                    decoded_label.append(EOS)
                    break
                else:
                    decoded_words.append(converter.decode(ni))
                    decoded_label.append(ni)

            for pred, target in zip(decoded_label, target_variable[1:,:]):
                if pred == target:
                    n_correct += 1

            if i % 1000 == 0:
                print(i)
                texts = cpu_texts[0]
                print('pred:%-20s, gt: %-20s' % (decoded_words, texts))

    accuracy = n_correct / float(n_total)
    print('Test loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))
    return loss_avg.val(), accuracy


def trainBatch(encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, device, teach_forcing_prob=1):

    data = train_iter.next()
    cpu_images, cpu_texts = data
    cpu_images = cpu_images.to(device)
    b = cpu_images.size(0)
    target_variable = converter.encode(cpu_texts).to(device)
    #print(target_variable.shape)

    encoder_outputs = encoder(cpu_images)
    #print(encoder_outputs.size())
    decoder_input = target_variable[0].to(device)
    decoder_hidden = decoder.initHidden(b).to(device)
    loss = 0.0
    teach_forcing = True if random.random() > teach_forcing_prob else False

    if teach_forcing:
        for di in range(1, target_variable.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]
    else:
        for di in range(1, target_variable.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze()
            decoder_input = ni

    encoder.zero_grad()
    decoder.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss


if __name__ == '__main__':

    for epoch in range(opt.niter):
        train_iter = iter(train_loader)
        i = 0
        epoch_avg = utils.averager()

        while i < len(train_loader)-1:

            encoder.train()
            decoder.train()
            cost = trainBatch(encoder, decoder, criterion, encoder_optimizer, 
                              decoder_optimizer, device, teach_forcing_prob=opt.teaching_forcing_prob)
            loss_avg.add(cost)
            epoch_avg.add(cost)
            
            i += 1

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' % (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        # save checkpoint
        if epoch % opt.saveInterval == 0:
            loss, acc = val(encoder, decoder, criterion, val_loader, device)

            with open('learning_curves_attn.txt', 'a+') as f:
                print(epoch, epoch_avg.val().item(), loss.item(), acc, file=f)

            torch.save(encoder.state_dict(), '{0}/encoder_{1}.pth'.format(opt.experiment, epoch))
            torch.save(decoder.state_dict(), '{0}/decoder_{1}.pth'.format(opt.experiment, epoch))
