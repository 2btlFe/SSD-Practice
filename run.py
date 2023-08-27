import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import time
import pandas as pd
import wandb

from Dataset_DataLoader import make_datapath_list, Anno_xml2list, DataTransform, VOCDataset, od_collate_fn
from ssd_model_forwad import SSD
from loss_function import MultiBoxLoss

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:  # 바이어스 항이 있으면
            nn.init.constant(m.bias, 0.0)

#모델 학습시키는 함수 작성
def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):

    # GPU 사용가능한지 확인
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용 중인 장치:", device)

    #네트워크를 GPU로
    net.to(device)

    #네트워크가 어느 정도 고정되면 고속화시킨다
    torch.backends.cudnn.benchmark = True

    #반복자의 카운터 설정
    iteration = 1
    epoch_train_loss = 0.0  #에폭 손실 합
    epoch_val_loss = 0.0    #에폭 손실 합
    logs = []

    #에폭 루프
    for epoch in range(num_epochs+1):

        #시작 시간 저장
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epoch별 훈련 및 검증을 루프
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() # 모델을 훈련 모드로
                print(' (train) ')
            else:
                if((epoch+1)%10 == 0):  #10번당 1번만
                    net.eval()  # 모델을 검증 모드로
                    print('----------')
                    print(' (val) ')
                else:
                    # 검증은 10회에 1회만 실시
                    continue

            # 데이터 로더에서 미니 배치씩 꺼내 루프
            for images, targets in dataloader_dict[phase]:

                #GPU를 사용할 수 있으면 GPU에 데이터를 보낸다
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]   # 리스트 각 요소의 텐서를 GPU로

                #옵티마이저 초기화
                optimizer.zero_grad()

                #순전파 계산
                with torch.set_grad_enabled(phase == 'train'):
                    #순전파 계산
                    outputs = net(images)

                    #손실 계산
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    #훈련 시에는 역전파
                    if phase == 'train':
                        loss.backward() #경사 계산

                        # 경사가 너무 커지면 계산이 부정확해 clip에서 최대 경사 2.0에 고정
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)

                        optimizer.step()    #파라미터 갱신

                        if(iteration % 10 == 0):    #10iter에 한번 thstlf vytl
                            t_iter_finish = time.time()
                            duration = t_iter_finish-t_iter_start
                            print('반복 {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item(), duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item() #loss.item을 써서 숫자 타입을 만들어야 한다
                        iteration += 1
                    
                    #검증시
                    else:
                        epoch_val_loss += loss.item()

            #epoch의 phase 당 loss와 정답률
            t_epoch_finish = time.time()
            print('------------')
            print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss, epoch_val_loss))
            print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
            t_epoch_start = time.time()

            #로그를 저장
            wandb.log({"train_loss": epoch_train_loss, "val_loss": epoch_val_loss})
            log_epoch = {'epoch': epoch+1, 
                        'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv("log_output.csv")

            epoch_train_loss = 0.0  # epoch의 손실합
            epoch_val_loss = 0.0  # epoch의 손실합

            # 네트워크를 저장한다
            if((epoch+1) % 10 == 0):
                torch.save(net.state_dict(), 'weights/ssd300_' + str(epoch+1) + '.pth')

if __name__ == "__main__":
    #wandb 설정
    wandb.init(
    project="Pytorch_Advanced2 - SSD",
    entity="bc6817",
    name="SSD_Train",
    resume ="allow",
    mode="online",
    )

    wandb.run.name = 'SSD'
    wandb.run.save()

    cfg = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 25
    }

    wandb.config.update(cfg)

    
    #파일 경로 리스트 취득
    rootpath = "./data/VOCdevkit/VOC2012"

    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

    # 1.Dataset 작성
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    color_mean = (104, 117, 123)  # (BGR) 색상의 평균값
    input_size = 300  # 화상의 input 사이즈를 300×300으로

    train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(input_size, color_mean),
                               transform_anno=Anno_xml2list(voc_classes))
    
    val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val",transform=DataTransform(input_size, color_mean), 
                             transform_anno=Anno_xml2list(voc_classes))


    # 2.DataLoader 작성
    batch_size = 32
    
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
    
    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
    
    # 사전 오브젝트로 정리
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # 3. 네트워크 모델 작성
    # SSD300 설정
    ssd_cfg = {
        'num_classes': 21,  # 배경 클래스를 포함한 총 클래스 수
        'input_size': 300,  # 화상의 입력 크기
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 출력할 Box 화면비의 종류
        'feature_maps': [38, 19, 10, 5, 3, 1],  # 각 source의 화상 크기
        'steps': [8, 16, 32, 64, 100, 300],  # DBOX의 크기를 정한다
        'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOX의 크기를 정한다
        'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOX의 크기를 정한다
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

    # SSD 네트워크 모델
    net = SSD(phase="train", cfg=ssd_cfg)

    # SSD 초기 가중치 설정
    # ssd의 vgg에 가중치 로드
    vgg_weights = torch.load('./weights/vgg16_reducedfc.pth')
    net.vgg.load_state_dict(vgg_weights)

    # ssd의 기타 네트워크 가중치는 He의 초기치로 초기화
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)

    # GPU 사용가능한지 확인
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용 중인 장치:", device)

    print('네트워크 설정 완료: 학습된 가중치를 로드했습니다')


    # 4. 손실함수 설정
    criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)

    # 5. 최적화 기법 설정
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    # 6. 학습 및 검증 실시
    num_epochs = 25
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)