import cv2
import os
from dataLoader import MyDataSet, DDataSet
from detector import Detector
from GAN import Discriminator, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils import data
from torch.autograd import Variable
from tqdm import tqdm

import numpy as np


def train(model, trainloader, optimizer, criterion, epoch, device):
    if device == 'cuda':
        model = model.cuda()
        criterion = criterion.cuda()
    bar = tqdm(trainloader)
    bar.set_description(f'epoch {epoch:2}')
    for (X, y), dirname in bar:
        if device == 'cuda':
            X = X.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        pred = model.forward(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        if device == 'cuda':
            loss = loss.cpu()
        bar.set_postfix_str(f'loss={loss.item():.2f}')


def test(model, device, testloader, criterion):
    with torch.no_grad():
        if device == 'cuda':
            model = model.cuda()
            criterion = criterion.cuda()
        for (X, y), dirname in testloader:
            if device == 'cuda':
                X = X.cuda()
                y = y.cuda()
            pred = model.forward(X)
            loss = criterion(pred, y)

            if device == 'cuda':
                pred = pred.cpu()
                loss = loss.cpu()

            for i in range(len(pred)):
                print(dirname[i])
                origin = (pred[i].numpy() * 255).transpose((1, 2, 0)).astype(np.uint8)
                # print(dirname[i][:-4]+"_gen"+dirname[i][-4:])
                gtt = cv2.imread(dirname[i])
                cv2.imshow("truth", gtt)
                cv2.imshow("pred", origin)
                # cv2.imwrite(dirname[i][:-4]+"_gen"+dirname[i][-4:], origin)
                if cv2.waitKey() == 'q':
                    cv2.destroyAllWindows()
            print(f"loss:{loss}")


def query(model, img_list):
    image = 0
    for dirname in img_list:
        if type(image) == int:
            image = cv2.imread(dirname)
        else:
            image = np.append(image, cv2.imread(dirname), axis=2)
    inputs = torchvision.transforms.F.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        pred = model.forward(inputs)
        pred = (pred[0].numpy() * 255).transpose((1, 2, 0)).astype(np.uint8)
        cv2.imshow("pred", pred)
        strkey = cv2.waitKey()
        print(strkey)
        while True:
            if strkey == 113 or strkey == 81:  # q or Q
                # print("join!")
                cv2.destroyAllWindows()
                image = np.append(image[:, :, 3:], pred, axis=2)
                inputs = torchvision.transforms.F.to_tensor(image).unsqueeze(0)
                pred = model.forward(inputs)
                pred = (pred[0].numpy() * 255).transpose((1, 2, 0)).astype(np.uint8)
                cv2.imshow("pred", pred)
                strkey = cv2.waitKey()
            else:
                break
        cv2.destroyAllWindows()


def cnn_main():
    join_num = 3
    batch = 5
    epoch = 5
    lr = 5e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Generator(join_num=join_num)
    version = "v0gan_3"

    pkl_file = 'checkpoints\\'+version
    if not os.path.exists(pkl_file):
        os.makedirs(pkl_file)
        file = open("version_info.txt", 'a')
        file.writelines(version)
        file.write(str(model))
        file.close()
    para_save_road = pkl_file+'\\parameter_jn'+str(join_num)+version+'.pkl'

    if os.path.exists(para_save_road):
        model.load_state_dict(torch.load(para_save_road))

    trainloader = data.DataLoader(MyDataSet(root="data", mode="train", join_num=join_num), batch_size=batch,
                                  shuffle=True, num_workers=0)
    testloader = data.DataLoader(MyDataSet(root="test_data", mode="test", join_num=join_num), batch_size=1,
                                 num_workers=0)

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # for i in range(epoch):
    #     train(model, trainloader, optimizer, criterion, i, device)
    #     torch.save(model.state_dict(), para_save_road)
    #     print("result saved")
    # test(model, device, testloader, criterion)

    img_list = ['test_data\\1622380577.5852423_0\\8.jpg',
                'test_data\\1622380577.5852423_0\\9.jpg',
                'test_data\\1622380577.5852423_0\\10.jpg']
    query(model, img_list)
    pass


def GAN_test(D, G, D_test_loader, device):
    with torch.no_grad():
        if device == "cuda":
            D = D.cuda()
        for (X, y), dir_name in D_test_loader:
            if device == "cuda":
                X = X.cuda()
                y = y.cuda()
            pred = D(y, X)  # X:b*3jn*w*h
            gen_img = G(X)
            pred_gen = D(gen_img, X)
            if device == "cuda":
                pred = pred.cpu()
                pred_gen = pred_gen.cpu()
                gen_img = gen_img.cpu()
            for i in range(len(pred)):
                print(f"real_D:{pred[i]}")
                print(f"fake_D:{pred_gen}")
                origin = (gen_img[i].numpy() * 255).transpose((1, 2, 0)).astype(np.uint8)
                gt = cv2.imread(dir_name[i])
                cv2.imshow("show", gt)
                cv2.imshow("gen", origin)
                if cv2.waitKey() == 81:
                    return 0


def gan_train(D, G, D_optimizer, G_optimizer, criterion, D_dataloader, G_dataloader, device, epoch, batch):
    delta = 0.
    real_label = torch.tensor([[1-delta, 0+delta]])
    fake_label = torch.tensor([[0-delta, 1+delta]])
    if device == "cuda":
        D = D.cuda()
        G = G.cuda()
        criterion = criterion.cuda()
    # D training
    while True:
        flag = 0
        bar = tqdm(D_dataloader)
        bar.set_description(f'epoch {epoch:2}')
        for (X, real_img), dirname in bar:
            if device == 'cuda':
                X = X.cuda()
                real_img = real_img.cuda()
                real_label = Variable(real_label).cuda()
                fake_label = Variable(fake_label).cuda()
            fake_img = G(X).detach()
            D_real_pred = D(real_img, X)
            D_fake_pred = D(fake_img, X)
            real_loss = criterion(D_real_pred, real_label)
            fake_loss = criterion(D_fake_pred, fake_label)
            loss = real_loss + fake_loss
            D_optimizer.zero_grad()
            loss.backward()
            D_optimizer.step()
            if device == 'cuda':
                loss = loss.cpu()
                real_loss = real_loss.cpu()
                fake_loss = fake_loss.cpu()
            if loss.item() < 0.01:
                flag += 1
                # if flag > 200:
                #     break
            bar.set_postfix_str(f'loss={loss.item():.5f} fl={fake_loss.item():.5f} rl={real_loss.item():.5f}')
        if flag > 20:
            break
    # G training
    while True:
        flag = 0
        G_bar = tqdm(G_dataloader)
        G_bar.set_description(f'epoch {epoch:2} G_train')
        for (X, real_img), dirname in G_bar:
            if device == 'cuda':
                X = X.cuda()
            fake_img = G(X)
            D_fake_img = D(fake_img, X)
            g_loss = criterion(D_fake_img, real_label)
            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()
            if device == 'cuda':
                g_loss = g_loss.cpu()
            if g_loss.item() < 0.5:
                flag += 1
                if flag > 10:
                    break
            G_bar.set_postfix_str(f'g_loss={g_loss.item():.7f}')
        if flag > 10:
            break


def gan_main():
    join_num = 3
    batch = 1
    epoch = 30
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = "v0gan_3"
    G = Generator(join_num=join_num)
    D = Discriminator(join_num=join_num)

    pkl_file = 'checkpoints\\' + version
    if not os.path.exists(pkl_file):
        os.makedirs(pkl_file)
        file = open("version_info.txt", 'a')
        file.writelines(version)
        file.write("\n")
        file.write(str(G))
        file.write("\n")
        file.write(str(D))
        file.write("\n")
        file.close()
    G_para_save_road = pkl_file + '\\Gparameter_jn' + str(join_num) + version + '.pkl'
    D_para_save_road = pkl_file + '\\Dparameter_jn' + str(join_num) + version + '.pkl'

    if os.path.exists(G_para_save_road):
        G.load_state_dict(torch.load(G_para_save_road))

    if os.path.exists(D_para_save_road):
        D.load_state_dict(torch.load(D_para_save_road))

    D_loader = data.DataLoader(MyDataSet(root="GANdata", mode="train", join_num=join_num), batch_size=batch,
                               shuffle=True, num_workers=0)
    G_loader = data.DataLoader(MyDataSet(root="GANdata", mode="train", join_num=join_num), batch_size=1,
                               shuffle=True, num_workers=0)
    D_test_loader = data.DataLoader(DDataSet(root="test_data", join_num=join_num), batch_size=1, num_workers=0)

    criterion = nn.BCELoss()

    for i in range(epoch):
        D_optimizer = torch.optim.Adam(D.parameters(), lr=lr*(0.9**(i//2)))
        G_optimizer = torch.optim.SGD(G.parameters(), lr=3*lr*(0.9**(i//10)), momentum=0.5)
        gan_train(D, G, D_optimizer, G_optimizer, criterion, D_dataloader=D_loader, G_dataloader=G_loader,
                  device=device, epoch=i, batch=batch)
        torch.save(D.state_dict(), D_para_save_road)
        torch.save(G.state_dict(), G_para_save_road)
        print("result saved")
    GAN_test(D, G, D_test_loader, device)
    # testloader = data.DataLoader(MyDataSet(root="test_data", mode="test", join_num=join_num), batch_size=1,
    #                              num_workers=0)
    # test(G, device, testloader, criterion)


if __name__ == "__main__":
    cnn_main()
