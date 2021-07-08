from __future__ import unicode_literals, print_function, division
import copy
import sys
from io import open
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')

SOS_token = 0
EOS_token = 1001
hidden_size = 300
epochs = 35 + 1
currentEpoch = 0
input_size = 102
output_size = 1021
MAX_LENGTH = 79

teacher_forcing_ratio = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_sequences = []
target_sequences = []

encoder_model_save_path = "C:\\Users\\Mrak\\PycharmProjects\\rnn_syntax_error_correction\\encoderModel"
decoder_model_save_path = "C:\\Users\\Mrak\\PycharmProjects\\rnn_syntax_error_correction\\decoderModel"

training_sequences_source_file = "_trainingSourceFile_all_files_12000"
validation_sequences_source_file = "_validationSourceFile-2000"

trainingSequencesFile = open(validation_sequences_source_file, "r")
validationSequencesFile = open(validation_sequences_source_file, "r")

lossesInTrainingFile = open("_average_loses_in_training", "a")
lossesInValidationFile = open("_average_loses_in_validation", "a")

trainingLines = trainingSequencesFile.read().splitlines()
validationLines = validationSequencesFile.read().splitlines()

def createInputAndOutputSequences(source="training"):
    lines = None
    if source == "training":
        lines = trainingLines
    elif source == "validation":
        lines = validationLines

    global count
    random.shuffle(lines)
    for line in lines:
        # every line has 80 tokens(100)
        lineTokens = line.split()

        if (len(lineTokens) < 80):
            continue

        # random_number = random.randint(0, 10)
        # parseUntilToken = 55 + random_number

        parseUntilToken = 79

        count = 0
        source_sequence_single = []
        target_sequence_single = []
        for token in lineTokens:
            token = int(token)
            count += 1

            if count == 80:
                target_sequence_single.append(token)
                input_sequences.append(source_sequence_single)
                target_sequences.append(target_sequence_single)
            elif count <= parseUntilToken:
                source_sequence_single.append(token)

createInputAndOutputSequences()
print(trainingLines[0])
print(input_sequences[0])
print(target_sequences[0])
print(trainingLines[101])
print(input_sequences[101])
print(target_sequences[101])

#not added EOS and SOS TOKEN

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 2

        self.embedding = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=2)
        # self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden_state):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded

        output, hidden_state = self.gru(output, hidden_state)

        return output, hidden_state

    #create initial hidden state
    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.3, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=2)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden_state, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden_state[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden_state = self.gru(output, hidden_state)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden_state, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden_state = encoder.initHidden()

    #clear optimizer from previous iterations
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #number of words in tensor(sequence)(100)
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    #initial encoder outputs(100)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    #iterate through all words of input tensor (1 sentence)
    for ei in range(input_length):
        encoder_output, encoder_hidden_state = encoder(
            input_tensor[ei], encoder_hidden_state)

        #add encoder output of one pass through encoder rnn to list of all encoder outuputs
        encoder_outputs[ei] = encoder_output[0, 0]

    #decoder input contains only initial token
    decoder_input = torch.tensor([[SOS_token]], device=device)

    #encoder hidden (last) from encoder rnn-a is first decoder hidden
    decoder_hidden_state = encoder_hidden_state

    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = True

    loss = 0
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden_state, decoder_attention = decoder(
                decoder_input, decoder_hidden_state, encoder_outputs)

            loss += criterion(decoder_output, target_tensor[di])
            #next input to decoder rnn is word from target_tensor
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden_state, decoder_attention = decoder(
                decoder_input, decoder_hidden_state, encoder_outputs)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    #does backward propagation
    loss.backward()

    #call for updating weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, learning_rate, trainingOrValidation):
    start = time.time()

    numberOfSequences = 0
    if trainingOrValidation == "training":
        numberOfSequences = 12001
    elif trainingOrValidation == "validation":
        numberOfSequences = 2001

    print_loss_total = 0
    print_loss_total_to_file = 0

    #set optimizer
    #this was mentioned in work that was used
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    input_tensor_sequences = []
    target_tensor_sequences = []
    #create input i target tensor seqeunces - 12001 of them
    for i in range(0, numberOfSequences):
        input_tensor_sequence = torch.tensor(input_sequences[i], dtype=torch.long, device=device).view(-1, 1)
        target_tensor_sequence = torch.tensor(target_sequences[i], dtype=torch.long, device=device).view(-1, 1)
        input_tensor_sequences.append(input_tensor_sequence)
        target_tensor_sequences.append(target_tensor_sequence)

    criterion = nn.NLLLoss()

    #iterates through all sequences
    for i in range(1, numberOfSequences):
        input_tensor_sequence = input_tensor_sequences[i]
        target_tensor_sequence = target_tensor_sequences[i]

        loss = train(input_tensor_sequence, target_tensor_sequence, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

        print("Current epoch: " + str(currentEpoch) + "  " + trainingOrValidation)
        print("Trained sequence: " + str(i))
        #loss from every iteration is added to sum
        print_loss_total += loss
        print_loss_total_to_file += loss

        if trainingOrValidation == "training":
            if (i + 1) % 4000 == 0:
                print_loss_total_to_file_average = print_loss_total_to_file / 4000
                print_loss_total_to_file = 0
                lossesInTrainingFile.write("Epoch: " + str(currentEpoch) + ". Average 4000 loss:\n" + str(print_loss_total_to_file_average) + "\n")
                lossesInTrainingFile.flush()
        elif trainingOrValidation == "validation":
            if (i + 1) % 600 == 0:
                print_loss_total_to_file_average = print_loss_total_to_file / 600
                print_loss_total_to_file = 0
                lossesInValidationFile.write("Epoch: " + str(currentEpoch) + ". Average 600 loss:\n" +  str(print_loss_total_to_file_average) + "\n")
                lossesInValidationFile.flush()

        #every 100 sequences pring loss
        print_every = 100
        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, i / 12001),
                                         i, i / 12001 * 100, print_loss_avg))

#input_size - vocabulary
encoder1 = EncoderRNN(input_size, hidden_size).to(device)

#output_size - vocabulary
attn_decoder1 = AttnDecoderRNN(hidden_size, output_size, dropout_p=0.3).to(device)

for i in range(1, epochs):
    currentEpoch = i

    learning_rate = 1
    if i >= 9:
        learning_rate = (1/(1 + 0.5 * (i - 9))) * 1
        print("Learning rate: " + str(learning_rate))

    input_sequences.clear()
    target_sequences.clear()
    createInputAndOutputSequences("training")

    encoderCopy = copy.deepcopy(encoder1)

    trainIters(encoder1, attn_decoder1, learning_rate, "training")

    torch.save(encoder1, encoder_model_save_path + str(currentEpoch))
    torch.save(attn_decoder1, decoder_model_save_path + str(currentEpoch))

    input_sequences.clear()
    target_sequences.clear()
    createInputAndOutputSequences("validation")

    encoderCopy = copy.deepcopy(encoder1)
    decoderCopy = copy.deepcopy(attn_decoder1)

    trainIters(encoderCopy, decoderCopy, learning_rate, "validation")