###################################################################################################
###################################################################################################
#################################        Adaptive Codebook        #################################
###################################################################################################
###################################################################################################

''' Here the implementation of Learning Multi-Rate Vector Quantization for Remote Deep Inference by
May Malka, Shai Ginzach, and Nir Shlezinger

For further questions: maymal@post.bgu.ac.il
'''

###################################################################################################
###################################################################################################
#################################             Imports             #################################
###################################################################################################
###################################################################################################
import random
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import ssl
import torch.nn as nn
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import time
import pickle
# from modulation import QAM, PSK
# import matplotlib.pyplot as plt
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################################################################################
###################################################################################################
#################################   Globals & Hyperparameters     #################################
###################################################################################################
###################################################################################################

BATCH_SIZE = 1
LEARNING_RATE = 3e-4
EPOCHS = 20
NUM_CLASSES = 10       # Classification of CIFAR100
NUM_EMBED = 256  # Number of vectors in the codebook.
snr = 12

###################################################################################################
###################################################################################################
#################################         Data Arrangment         #################################
###################################################################################################
###################################################################################################

# def get_test_transforms():
#     test_transform = transforms.Compose(
#                 [transforms.ToTensor(),
#       		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#     return test_transform
#
# def get_train_transforms():
#     transform = transforms.Compose(
#                 [transforms.RandomCrop(32, padding=4),
#                  transforms.RandomHorizontalFlip(),
#                  transforms.ToTensor(),
#                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#     return transform
#
# train_transform = get_train_transforms()
# test_transform = get_test_transforms()
#
# ssl._create_default_https_context = ssl._create_unverified_context
# path = "/tmp/cifar10"
# trainset = datasets.CIFAR10(root = path, train=True, download=True, transform=train_transform)
# testset = datasets.CIFAR10(root = path, train=False, download=True, transform=test_transform)
#
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# Dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
path = "/tmp/cifar10"
train = torchvision.datasets.CIFAR10(root = path, train=True, download=False, transform=transform_train)
test = torchvision.datasets.CIFAR10(root = path, train=False, download=False, transform=transform_test)
# print(len(train))
# print(len(test))
dataset_train1_spl, dataset_train2_spl, _ = torch.utils.data.random_split(train, [25000, 25000, len(train) - 50000])
dataset_test_spl, _ = torch.utils.data.random_split(test, [10000, len(test) - 10000])
# dataset_test_spl = test
# test_data_loader = torch.utils.data.DataLoader(dataset_test_spl, batch_size=1000, shuffle=False, num_workers=2)  # 用于测试,batch_size原值为1000

trainloader = torch.utils.data.DataLoader(dataset_train1_spl, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset_test_spl, batch_size=1, shuffle=False)

###################################################################################################
###################################################################################################
#################################       Adaptive Quantizer        #################################
###################################################################################################
###################################################################################################

class AdaptiveVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, codebook_size, lambda_c=0.1, lambda_p=0.33):
        super(AdaptiveVectorQuantizer, self).__init__()

        self.d = num_embeddings  # The size of the vectors
        self.p = codebook_size  # Number of vectors in the codebook

        # initialize the codebook
        self.codebook = nn.Embedding(self.p, self.d)
        self.codebook.weight.data.uniform_(-1 / self.p, 1 / self.p)

        # Balancing parameter lambda for the commintment loss
        self.lambda_c = lambda_c
        self.lambda_p = lambda_p

        self.first = True

    def trans_matrix(self, diagonal_value, size):
        # 创建一个形状为 size x size 的全零矩阵
        matrix = np.zeros((size, size))
        # 使用 fill_diagonal() 方法将对角线元素设置为 diagonal_value
        np.fill_diagonal(matrix, diagonal_value)

        other_value = (1 - diagonal_value) / (size - 1)
        # 将其他元素设置为 other_value
        matrix[np.where(matrix == 0)] = other_value
        # 控制矩阵元素精度
        matrix = matrix.astype(np.float32)
        # 将numpy转为tensor
        tensor_matrix = torch.from_numpy(matrix)
        return tensor_matrix

    def forward(self, inputs, num_vectors, prev_vecs, correct_p):
        # convert inputs from BCHW -> BHWC  # torch.Size([64, 32, 16, 16])
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape  # torch.Size[64,16,16,32]

        # Flatten input
        flat_input = inputs.view(-1, self.d)  # Tensor{262144,2}

        quant_vecs = []
        losses = []

        for num_actives in range(int(np.log2(num_vectors))):
            actives = self.codebook.weight[:pow(2, num_actives + 1)]

            # Calculate distances
            distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                         + torch.sum(actives ** 2, dim=1)
                         - 2 * torch.matmul(flat_input, actives.t()))    # Tensor{262144,2}

            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # Tensor{262144,1}
            encodings = torch.zeros(encoding_indices.shape[0], self.p, device=inputs.device)   # Tensor{262144,256}
            encodings.scatter_(1, encoding_indices, 1)

            trans_matrix = self.trans_matrix(correct_p, self.p).to(encodings.device)
            transmitted_encodings = torch.zeros_like(encodings)
            encodings_t = encodings.t()
            transmitted_encodings_t = transmitted_encodings.t()
            for i in range(len(encodings_t)):
                p = random.random()
                if p <= trans_matrix[i][i]:
                    transmitted_encodings_t[i] = encodings_t[i]
                else:
                    integer = int((p - trans_matrix[i][i]) // ((1 - correct_p) / self.p))
                    transmitted_encodings_t[i] = (encodings_t[(i + 1 + integer) % len(encodings_t)])

            encodings_t=transmitted_encodings_t
            encodings = encodings_t.t()
            # Quantize and unflatten
            quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
            quant_vecs.append(quantized)  # torch.Size[64,16,16,32]

        for num_actives in range(int(np.log2(num_vectors))):
            if self.training:
                # Loss
                q_latent_loss = F.mse_loss(quant_vecs[num_actives], inputs.detach())  # commitment loss

                if num_actives == 0:
                    prox_loss = 0
                    e_latent_loss = F.mse_loss(quant_vecs[num_actives].detach(), inputs)  # alignment loss

                elif num_actives == 1:
                    e_latent_loss = F.mse_loss(quant_vecs[num_actives].detach(), inputs)
                    prox_loss = (num_actives * self.lambda_p) * F.mse_loss(prev_vecs[:pow(2, num_actives + 1) // 2],
                                                                           actives[:pow(2, num_actives + 1) // 2])
                else:
                    e_latent_loss = 0
                    prox_loss = self.lambda_p * F.mse_loss(prev_vecs[:pow(2, num_actives + 1) // 2],
                                                           actives[:pow(2, num_actives + 1) // 2])  # proximity_loss

                cb_loss = q_latent_loss + self.lambda_c * e_latent_loss + prox_loss  # codebook loss

                quant_vecs[num_actives] = inputs + (quant_vecs[num_actives] - inputs).detach()  # gradient copying
                quant_vecs[num_actives] = quant_vecs[num_actives].permute(0, 3, 1, 2).contiguous()

            else:
                # convert quantized from BHWC -> BCHW
                # quant_vecs[num_actives] = quant_vecs[num_actives].permute(0, 3, 1, 2).contiguous()
                cb_loss = 0

            losses.append(cb_loss)

        return quant_vecs, losses, actives


###################################################################################################
###################################################################################################
########################################       Model        #######################################
###################################################################################################
###################################################################################################

class AdapCB_Model(nn.Module):
    def __init__(self, num_embeddings, codebook_size, lambda_c=0.1, lambda_p=0.33, quant=True):
        super(AdapCB_Model, self).__init__()

        self.encoder, self.decoder, self.classifier = self.split_net()
        self.quantizer = AdaptiveVectorQuantizer(num_embeddings, codebook_size, lambda_c, lambda_p)
        self.quant = quant
        self.linear = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.Sigmoid()  # 正式得还回去
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
    def build_model(self, pretrained=True, fine_tune=True):
        if pretrained:
            print('[INFO]: Loading pre-trained weights')
        elif not pretrained:
            print('[INFO]: Not loading pre-trained weights')
        inverted_residual_setting=[[1, 16, 1, 1],[6, 24, 2, 1],[6, 32, 3, 1],[6, 64, 4, 2],[6, 96, 3, 1],[6, 160, 3, 1],[6, 320, 1, 1]]
        model = models.mobilenet_v2(pretrained=pretrained, num_classes=1000,width_mult=1,inverted_residual_setting=inverted_residual_setting)
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in model.parameters():
                params.requires_grad = False

        # change the final classification head, it is trainable,
        model.dropout = nn.Dropout(0.2,inplace=True)
        model.fc = nn.Linear(in_features=1280*8*8, out_features=NUM_CLASSES, bias=True)
        return model

    def split_net(self):
        mobilenetv2 = self.build_model()

        encoder = []
        decoder = []
        classifier = []

        res_stop = 5
        for layer_idx, l in enumerate(mobilenetv2.features):
            if layer_idx <= res_stop:
                encoder.append(l)
            else:
                decoder.append(l)

        classifier.append(mobilenetv2.dropout)
        classifier.append(mobilenetv2.fc)

        Encoder = nn.Sequential(*encoder)
        Decoder = nn.Sequential(*decoder)
        Classifier = nn.Sequential(*classifier)
        return Encoder, Decoder, Classifier

    def get_accuracy(self, gt, preds):
        pred_vals = torch.max(preds.data, 1)[1]
        batch_correct = (pred_vals == gt).sum()
        return batch_correct

    def normalize(self,inputs):
        # Calculate the vector's magnitude
        mean = inputs.mean()
        output = inputs - mean
        return output

    def forward(self, inputs, num_active, prev_vecs, correct_p):
        z_e = self.encoder(inputs)
        x = torch.reshape(z_e, (z_e.size()[0], 32 * 16 * 16))
        line = self.linear(x)
        z_e1 = self.normalize(line)
        if self.quant == True:
            z_q, vq_loss, actives = self.quantizer(z_e1, num_active, prev_vecs, correct_p)
        else:
            z_q, vq_loss, actives = [z_e1], [0], None

        preds_list = []
        for vecs in range(len(z_q)):
            z_q_actives = z_q[vecs]  # [64, 32, 16, 16]
            x1 = torch.reshape(z_q_actives, (-1, 64, 4, 4))
            x1 = self.layer1(x1)
            z_q_actives = self.layer2(x1)
            preds_list.append(self.decoder(z_q_actives))   # [64, 1280, 8, 8]
            preds_list[vecs] = preds_list[vecs].reshape(preds_list[vecs].shape[0],
                                            preds_list[vecs].shape[1]*preds_list[vecs].shape[2]*preds_list[vecs].shape[3])  # [64, 81920]
            preds_list[vecs] = self.classifier(preds_list[vecs])
        return preds_list, vq_loss, actives, z_e


###################################################################################################
###################################################################################################
##################################       Testing Function        #################################
###################################################################################################
###################################################################################################


def test(model, num_active, criterion, prev_vecs=None, commitment=0.2):
    test_losses = []
    val_losses = 0
    test_accP1 = [0, 0, 0, 0, 0, 0, 0, 0]
    test_accP2 = [0, 0, 0, 0, 0, 0, 0, 0]
    test_accP3 = [0, 0, 0, 0, 0, 0, 0, 0]
    test_accP4 = [0, 0, 0, 0, 0, 0, 0, 0]
    test_accP5 = [0, 0, 0, 0, 0, 0, 0, 0]
    for epc in range(1, 51):
        print('This is {}-th epoch'.format(epc))

        test_acc1 = [0] * int(np.log2(num_active))
        test_acc2 = [0] * int(np.log2(num_active))
        test_acc3 = [0] * int(np.log2(num_active))
        test_acc4 = [0] * int(np.log2(num_active))
        test_acc5 = [0] * int(np.log2(num_active))
        model.eval()
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(testloader):
                start_time1 = time.time()
                # Apply the model
                correct_p = 1-0.1
                b += 1
                batch = X_test.to(device)
                val_preds, vq_val_loss, _, _ = model(batch, num_active, prev_vecs, correct_p)
                duration = time.time() - start_time1
                print("测试时间：", duration)
                loss_levels_val = []
                for q_level in range(len(val_preds)):
                    ce_val_loss = criterion(val_preds[q_level], y_test.to(device))
                    level_loss = ce_val_loss + commitment * vq_val_loss[q_level]
                    loss_levels_val.append(level_loss)
                    test_acc1[q_level] += model.get_accuracy(y_test.to(device), val_preds[q_level])
                val_loss = sum(loss_levels_val)
                val_losses += val_loss.item()

            test_losses.append(val_losses / b)
            # total_train_acc = [100 * (acc.item()) / len(trainset) for acc in train_acc]
            total_test_acc1 = [100 * (acc.item()) / len(testset) for acc in test_acc1]
            # print(f'Train Models Accuracy at epoch {epc + 1} is {total_train_acc}%')
            print(f'Test1 Models Accuracy at epoch {epc} is {total_test_acc1}%')
            test_accP1 = [x + y for x, y in zip(test_accP1, total_test_acc1)]
            print(f'test_accP1 is {test_accP1}%')


            for b, (X_test, y_test) in enumerate(testloader):
                # Apply the model
                correct_p = 1-0.05
                b += 1
                batch = X_test.to(device)
                val_preds, vq_val_loss, _, _ = model(batch, num_active, prev_vecs, correct_p)
                loss_levels_val = []
                for q_level in range(len(val_preds)):
                    ce_val_loss = criterion(val_preds[q_level], y_test.to(device))
                    level_loss = ce_val_loss + commitment * vq_val_loss[q_level]
                    loss_levels_val.append(level_loss)
                    test_acc2[q_level] += model.get_accuracy(y_test.to(device), val_preds[q_level])
                val_loss = sum(loss_levels_val)
                val_losses += val_loss.item()
            test_losses.append(val_losses / b)
            # total_train_acc = [100 * (acc.item()) / len(trainset) for acc in train_acc]
            total_test_acc2 = [100 * (acc.item()) / len(testset) for acc in test_acc2]
            # print(f'Train Models Accuracy at epoch {epc + 1} is {total_train_acc}%')
            print(f'Test2 Models Accuracy at epoch {epc} is {total_test_acc2}%')
            test_accP2 = [x + y for x, y in zip(test_accP2, total_test_acc2)]
            print(f'test_accP2 is {test_accP2}%')

            for b, (X_test, y_test) in enumerate(testloader):
                # Apply the model
                correct_p = 1-0.01
                b += 1
                batch = X_test.to(device)
                val_preds, vq_val_loss, _, _ = model(batch, num_active, prev_vecs, correct_p)
                loss_levels_val = []
                for q_level in range(len(val_preds)):
                    ce_val_loss = criterion(val_preds[q_level], y_test.to(device))
                    level_loss = ce_val_loss + commitment * vq_val_loss[q_level]
                    loss_levels_val.append(level_loss)
                    test_acc3[q_level] += model.get_accuracy(y_test.to(device), val_preds[q_level])
                val_loss = sum(loss_levels_val)
                val_losses += val_loss.item()
            test_losses.append(val_losses / b)
            # total_train_acc = [100 * (acc.item()) / len(trainset) for acc in train_acc]
            total_test_acc3 = [100 * (acc.item()) / len(testset) for acc in test_acc3]
            # print(f'Train Models Accuracy at epoch {epc + 1} is {total_train_acc}%')
            print(f'Test3 Models Accuracy at epoch {epc} is {total_test_acc3}%')
            test_accP3 = [x + y for x, y in zip(test_accP3, total_test_acc3)]
            print(f'test_accP3 is {test_accP3}%')

            for b, (X_test, y_test) in enumerate(testloader):
                # Apply the model
                correct_p = 1-0.005
                b += 1
                batch = X_test.to(device)
                val_preds, vq_val_loss, _, _ = model(batch, num_active, prev_vecs, correct_p)
                loss_levels_val = []
                for q_level in range(len(val_preds)):
                    ce_val_loss = criterion(val_preds[q_level], y_test.to(device))
                    level_loss = ce_val_loss + commitment * vq_val_loss[q_level]
                    loss_levels_val.append(level_loss)
                    test_acc4[q_level] += model.get_accuracy(y_test.to(device), val_preds[q_level])
                val_loss = sum(loss_levels_val)
                val_losses += val_loss.item()
            test_losses.append(val_losses / b)
            # total_train_acc = [100 * (acc.item()) / len(trainset) for acc in train_acc]
            total_test_acc4 = [100 * (acc.item()) / len(testset) for acc in test_acc4]
            # print(f'Train Models Accuracy at epoch {epc + 1} is {total_train_acc}%')
            print(f'Test4 Models Accuracy at epoch {epc} is {total_test_acc4}%')
            test_accP4 = [x + y for x, y in zip(test_accP4, total_test_acc4)]
            print(f'test_accP4 is {test_accP4}%')

            for b, (X_test, y_test) in enumerate(testloader):
                # Apply the model
                correct_p = 1-0.001
                b += 1
                batch = X_test.to(device)
                val_preds, vq_val_loss, _, _ = model(batch, num_active, prev_vecs, correct_p)
                loss_levels_val = []
                for q_level in range(len(val_preds)):
                    ce_val_loss = criterion(val_preds[q_level], y_test.to(device))
                    level_loss = ce_val_loss + commitment * vq_val_loss[q_level]
                    loss_levels_val.append(level_loss)
                    test_acc5[q_level] += model.get_accuracy(y_test.to(device), val_preds[q_level])
                val_loss = sum(loss_levels_val)
                val_losses += val_loss.item()
            test_losses.append(val_losses / b)
            # total_train_acc = [100 * (acc.item()) / len(trainset) for acc in train_acc]
            total_test_acc5 = [100 * (acc.item()) / len(testset) for acc in test_acc5]
            # print(f'Train Models Accuracy at epoch {epc + 1} is {total_train_acc}%')
            print(f'Test5 Models Accuracy at epoch {epc} is {total_test_acc5}%')
            test_accP5 = [x + y for x, y in zip(test_accP5, total_test_acc5)]
            print(f'test_accP5 is {test_accP5}%')

    return test_accP1, test_accP2, test_accP3, test_accP4, test_accP5

###################################################################################################
###################################################################################################
##################################           Helpers              #################################
###################################################################################################
###################################################################################################

def scatter(array,train_points):
    plt.rcParams["figure.figsize"] = (10,10)
    colors = ['red', 'green', 'blue', 'purple', 'orange','magenta','cyan','yellow']
    for i,level in enumerate(array):
        x = np.array([elem[0] for elem in level.detach().cpu()])
        y = np.array([elem[1] for elem in level.detach().cpu()])
        name =  str(2*pow(2,i)) + 'Bits Vectors'
        train_points_level = train_points[-1]
        train_points_level = train_points_level[:20000]
        train_x_vals = np.array([elem[0] for elem in train_points_level.detach().cpu()])
        train_y_vals = np.array([elem[1] for elem in train_points_level.detach().cpu()])
        plt.scatter(train_x_vals, train_y_vals,s=10, alpha=0.1,label = 'Train Vectors')
        plt.scatter(x, y,s=250, alpha=1,label = name,c=colors[i % 8])

        # Add axis labels and a title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Scatter Plot')
        plt.grid()
        plt.legend(loc='best')
        # Show the plot
        plt.show()

def init_weights(m):
    if type(m) == nn.Embedding:
        torch.nn.init.normal_(m.weight, mean=0, std=1)


###################################################################################################
###################################################################################################
################################      Train Adaptive Codebook       ###############################
###################################################################################################
###################################################################################################


prev_vecs = None                      # Start with empty codebook
criterion = nn.CrossEntropyLoss()
# mod = QAM(NUM_EMBED, snr)
mobilenetv2_ac = AdapCB_Model(num_embeddings=2, codebook_size=NUM_EMBED, quant=True)
mobilenetv2_ac.quantizer.apply(init_weights)  # Initialize weights of codebook from normal distribution (0,1)
mobilenetv2_ac.to(device)
mobilenetv2_ac.load_state_dict(torch.load('./model/model_7_epoch_5.pth')['model'])
optimizer = torch.optim.Adam(mobilenetv2_ac.parameters(), lr=LEARNING_RATE)
samples_for_scatter = []
vecs_to_save = []
# EPOCHS = [10, 9, 8, 7, 6, 6, 6, 6]  # Set number of epochs for each training phase. [10,9,8,7,6,6,6,6]
# correct_p = 0.99
# for level in range(int(np.log2(NUM_EMBED))):
#     num_active = pow(2,level+1)        # Number of vectors-to-train in CB
#     correct_p = 0.99
#     curr_vecs, encoder_samples = train(mobilenetv2_ac, optimizer, num_active, criterion, prev_vecs, EPOCHS[level], correct_p)
#     samples_for_scatter.append(encoder_samples)
#     with open('./samples_for_scatter/samples_for_scatter_{}.pkl'.format(level + 1), 'wb') as file: pickle.dump(samples_for_scatter, file)
#     vecs_to_save.append(curr_vecs)
#     with open('./vecs_to_save/vecs_to_save_{}.pkl'.format(level + 1), 'wb') as file: pickle.dump(vecs_to_save, file)
#     prev_vecs = curr_vecs
#     # scatter(vecs_to_save,samples_for_scatter)
# EPOCHS = 10
num_active = pow(2, 8)
# for i in range(1, 50):
#     print('This is {}-th epoch'.format(i))
test_accP1, test_accP2, test_accP3, test_accP4, test_accP5 = test(mobilenetv2_ac, num_active, criterion, prev_vecs)
# print(f'TestP1 Models Accuracy is {test_accP1}%')
# print(f'TestP2 Models Accuracy is {test_accP2}%')
# print(f'TestP3 Models Accuracy is {test_accP3}%')
# print(f'TestP4 Models Accuracy is {test_accP4}%')
# print(f'TestP5 Models Accuracy is {test_accP5}%')

# T1 = [test_accP1[acc] / 100 for acc in test_accP1]
# T2 = [test_accP2[acc] / 100 for acc in test_accP2]
# T3 = [test_accP3[acc] / 100 for acc in test_accP3]
# T4 = [test_accP4[acc] / 100 for acc in test_accP4]
# T5 = [test_accP5[acc] / 100 for acc in test_accP5]
# print(f'TestP1 Models Accuracy is {T1}%')
# print(f'TestP2 Models Accuracy is {T2}%')
# print(f'TestP3 Models Accuracy is {T3}%')
# print(f'TestP4 Models Accuracy is {T4}%')
# print(f'TestP5 Models Accuracy is {T5}%')