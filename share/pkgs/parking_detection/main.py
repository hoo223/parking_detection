''' Module Import '''
import numpy as np
import matplotlib.pyplot as plt
import os, copy
from PIL import Image
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.models as models
import rexnetv1

parser = argparse.ArgumentParser("Set parking lot occupancy detection project parameters", add_help=False)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--model', type=str, default='AlexNet')
parser.add_argument('--learning_rate', type=float, default='0.0005')
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--pretrained', type=bool, default=False)
args = parser.parse_args()

class Data:
    def __init__(self, img_path, txt_path, transforms = None):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.img_list = [os.path.join(img_path, i.split()[0]) for i in lines]
            self.label_list = [i.split()[1] for i in lines]
            self.transforms = transforms
    
    def __getitem__(self, index):
        try:
            img_path = self.img_list[index]
            img = Image.open(img_path)
            img = self.transforms(img)
            label = self.label_list[index]
        except:
            return None
        return img, label
    
    def __len__(self):
        return len(self.label_list)

''' 불러온 특정 모델에 대해 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def train(model, train_loader, optimizer, log_interval):
    model.train() # 모델을 학습 상태로 지정
    for batch_idx, (image, label) in enumerate(train_loader):
        label = list(map(int, label))
        label = torch.Tensor(label)
        image = image.to(DEVICE) # 기존 정의한 장비에 할당
        label = label.to(DEVICE) # 기존 정의한 장비에 할당
        optimizer.zero_grad() # 기존 할당되어 있던 gradient 값 초기화
        output = model(image) # Forward propagation
        loss = criterion(output, label.long()) # loss 계산
        loss.backward() # Backpropagation을 통해 계산된 gradient 값을 각 파라미터에 할당
        optimizer.step() # 파라미터 업데이트
        
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f} ({})".format(
                  Epoch, batch_idx * len(image),
                  len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                  loss.item(),
                  datetime.now()))

''' 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def evaluate(model, test_loader):
    model.eval() # 모델을 평가 상태로 지정
    test_loss = 0
    correct = 0
    
    with torch.no_grad(): # 평가하는 단계에서 gradient를 통해 파라미터 값이 업데이트되는 현상을 방지
        for image, label in test_loader:
            label = list(map(int, label))
            label = torch.Tensor(label)
            image = image.to(DEVICE) # 기존 정의한 장비에 할당
            label = label.to(DEVICE) # 기존 정의한 장비에 할당
            output = model(image) # Forward propagation
            test_loss += criterion(output, label.long()).item() # loss 누적
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item() 
            
    test_loss /= len(test_loader.dataset) # 평균 loss 계산
    test_accuracy = 100. * correct / len(test_loader.dataset) # 정확도 계산
    return test_loss, test_accuracy

if __name__ == '__main__':
    ''' 딥러닝 모델을 설계할 때 활요하는 장비 확인 '''
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
        
    print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

    # Hyperparameter 설정
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    OPTIM = args.optim # SGD Adam
    MODEL = args.model # AlexNet ResNet101 ResNet50 ResNet34 ResNet18 RexNet
    PRETRAINED = args.pretrained

    ''' 이미지 데이터 불러오기(Train set, Test set)'''
    # preprocessing 정의
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])
    }

    img_path = '/root/share/datasets/Bluecom'
    trainset_txt = {'fold1': './splits/Custom_Paper/fold2345_all.txt',
                    'fold2': './splits/Custom_Paper/fold1345_all.txt',
                    'fold3': './splits/Custom_Paper/fold1245_all.txt',
                    'fold4': './splits/Custom_Paper/fold1235_all.txt',
                    'fold5': './splits/Custom_Paper/fold1234_all.txt'
                }
    testset_txt = {'fold1': './splits/Custom_Paper/fold1_all.txt',
                'fold2': './splits/Custom_Paper/fold2_all.txt',
                'fold3': './splits/Custom_Paper/fold3_all.txt',
                'fold4': './splits/Custom_Paper/fold4_all.txt',
                'fold5': './splits/Custom_Paper/fold5_all.txt',
                }

    train_loader = {}
    test_loader = {}

    if MODEL == 'RexNet':
        drop_last = True
    else:
        drop_last = False

    for i in range(len(trainset_txt)):
        train_dataset = Data(img_path, trainset_txt['fold'+str(i+1)], data_transforms['train'])
        test_dataset = Data(img_path, testset_txt['fold'+str(i+1)], data_transforms['val'])
        train_loader['fold'+str(i+1)] = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last)
        test_loader['fold'+str(i+1)] = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last)

    ''' 데이터 확인하기 (1) '''
    for (X_train, y_train) in train_loader['fold1']:
        print('X_train:', X_train.size(), 'type:', X_train.type())
        print('y_train:', len(y_train), 'type:', type(y_train))
        break

    # ''' 데이터 확인하기 (2) '''
    # pltsize = 1
    # plt.figure(figsize=(BATCH_SIZE * pltsize, pltsize))

    # for i in range(BATCH_SIZE):
    #     plt.subplot(1, BATCH_SIZE, i+1)
    #     plt.axis('off')
    #     plt.imshow(np.transpose(X_train[i], (1, 2, 0)))
    #     plt.title('Class: ' + str(y_train[i]))

    # 모델 불러오기
    if MODEL == 'AlexNet':
        model = models.alexnet(pretrained=PRETRAINED)
        model._modules['classifier']._modules['6'] = nn.Linear(4096, 2, bias=True)
    elif MODEL == 'ResNet18':
        model = models.resnet18(pretrained=PRETRAINED)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif MODEL == 'ResNet34':
        model = models.resnet34(pretrained=PRETRAINED)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif MODEL == 'ResNet50':
        model = models.resnet50(pretrained=PRETRAINED)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif MODEL == 'ResNet101':
        model = models.resnet101(pretrained=PRETRAINED)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif MODEL == 'RexNet': # BATCH_SIZE 1은 에러남
        model = rexnetv1.ReXNetV1(width_mult=1.0).cuda()
        if PRETRAINED:
            model.load_state_dict(torch.load('./saved_model/rexnetv1_1.0.pth'))
            
    init_model = copy.deepcopy(model)
    print(init_model)

    #for children in model.classifier.children():
    #    if isinstance(children, nn.Linear):
    #        if children.out_features == 1000:
    #            print(children)


    ''' 모델 훈련'''
    SAVE_PATH = './saved_model'
    LOG_PATH = './logs'
    if not os.path.isdir(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if not os.path.isdir(LOG_PATH):
        os.makedirs(LOG_PATH)

    for dataset in trainset_txt.keys():
        print("\n------------------    For ", dataset, " dataset    ------------------\n")
        
        # 기존에 훈련된 모델이 있는지 확인
        if PRETRAINED:
            MODEL_NAME = "{}_Pretrained_BS{}_{}_LR{}_EP{}_DS-{}".format(MODEL, BATCH_SIZE, OPTIM, str(LEARNING_RATE).split('.')[1], EPOCHS, dataset)
        else:
            MODEL_NAME = "{}_BS{}_{}_LR{}_EP{}_DS-{}".format(MODEL, BATCH_SIZE, OPTIM, str(LEARNING_RATE).split('.')[1], EPOCHS, dataset)
        BEST_MODEL_PATH = SAVE_PATH + "/" + MODEL_NAME + ".pt"
        if os.path.exists(BEST_MODEL_PATH):
            continue

        model = copy.deepcopy(init_model)
        model = model.cuda()
        
        if OPTIM == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        elif OPTIM == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()

        train_log = open(LOG_PATH+"/train_log_" + MODEL_NAME + ".txt", 'w')

        best_acc = 0
        best_ep = 0
        
        for Epoch in range(1, EPOCHS + 1):
            train(model, train_loader[dataset], optimizer, log_interval = 200)
            test_loss, test_accuracy = evaluate(model, test_loader[dataset])
            if test_accuracy > best_acc:
                best_acc = test_accuracy
                best_model = copy.deepcopy(model)
                best_ep = Epoch
                msg = "Best Model!\n"
                print(msg)
                train_log.writelines(msg)
            msg = "\nEPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".format(Epoch, test_loss, test_accuracy)
            print(msg)
            train_log.writelines(msg)

        torch.save(best_model.state_dict(), BEST_MODEL_PATH)
        msg = "\n------------------    Best Model at Epoch {} Saved    ------------------\n".format(best_ep)
        print(msg)
        train_log.writelines(msg)

        train_log.close()

    ''' 훈련된 모델 확인하기 '''
    test_log = open(LOG_PATH+"/test_log_" + MODEL_NAME + ".txt", 'w')
    test_log.writelines("Dataset List\n")
    for dataset in trainset_txt.keys():
        test_log.writelines("- {} train: {}, test: {} \n".format(dataset, trainset_txt[dataset], testset_txt[dataset]))

    msg = "\n*--------------------- Test Result ---------------------*\n"
    print(msg)
    test_log.writelines(msg)

    total_loss = 0
    for dataset in trainset_txt.keys():
        if PRETRAINED:
            MODEL_NAME = "{}_Pretrained_BS{}_{}_LR{}_EP{}_DS-{}".format(MODEL, BATCH_SIZE, OPTIM, str(LEARNING_RATE).split('.')[1], EPOCHS, dataset)
        else:
            MODEL_NAME = "{}_BS{}_{}_LR{}_EP{}_DS-{}".format(MODEL, BATCH_SIZE, OPTIM, str(LEARNING_RATE).split('.')[1], EPOCHS, dataset)
        BEST_MODEL_PATH = SAVE_PATH + "/" + MODEL_NAME + ".pt"
        
        if MODEL == 'AlexNet':
            trained_model = models.alexnet(pretrained=False)
            trained_model._modules['classifier']._modules['6'] = nn.Linear(4096, 2, bias=True)
        elif MODEL == 'ResNet18':
            trained_model = models.resnet18(num_classes=2, pretrained=False)
        elif MODEL == 'ResNet34':
            trained_model = models.resnet34(num_classes=2, pretrained=False)
        elif MODEL == 'ResNet50':
            trained_model = models.resnet50(num_classes=2, pretrained=False)
        elif MODEL == 'ResNet101':
            trained_model = models.resnet101(num_classes=2, pretrained=False)
        elif MODEL == 'RexNet': # BATCH_SIZE 1은 에러남
            trained_model = rexnetv1.ReXNetV1(width_mult=1.0).cuda()

        trained_model = trained_model.cuda()
        trained_model.load_state_dict(torch.load(BEST_MODEL_PATH))
        criterion = nn.CrossEntropyLoss()
        
        test_loss, test_accuracy = evaluate(trained_model, test_loader[dataset])
        total_loss += test_accuracy
        msg = '{} - loss: {}, acc: {}\n'.format(dataset, test_loss, test_accuracy)
        print(msg)
        test_log.writelines(msg)
        
    total_loss /= len(trainset_txt)
    msg = "average accuracy: {}".format(total_loss)
    print(msg)
    test_log.writelines(msg)

    test_log.close()