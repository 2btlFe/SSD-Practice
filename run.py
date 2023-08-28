import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import time
import pandas as pd
import wandb

# Custom
from Dataset_DataLoader import make_datapath_list, Anno_xml2list, DataTransform, VOCDataset, od_collate_fn
from ssd_model_forwad import SSD
from loss_function import MultiBoxLoss

# DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# argparse
import argparse

# os
import os

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--port', type=int, default=2033)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    return parser

def init_distributed_training(rank, opts):
    # 1. setting for distributed training
    opts.rank = rank
    opts.gpu = opts.rank % torch.cuda.device_count()
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)

    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))
    
    # 2. init_process_group
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:' + str(opts.port), 
                            world_size=opts.ngpus_per_node, 
                            rank=opts.rank)

    # if put this function, the all process block at all
    torch.distributed.barrier()

    # convert print fn iif rank is zero
    setup_for_distributed(opts.rank == 0)
    print('opts :', opts)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


'''
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 작업 그룹 초기화
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
'''


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:  # 바이어스 항이 있으면
            nn.init.constant_(m.bias, 0.0)

def main(rank, opts):
    #wandb 설정    
    wandb.init(
        project="SSD_with_DDP_16",
        entity="bc6817",
        name="SSD_Train",
        resume ="allow",
        mode="online",
        config=opts
    )
    wandb.run.name = 'SSD' + str(rank)

    #multi_gpu init
    init_distributed_training(rank, opts)
    local_gpu_id = opts.gpu
    
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
    # ddp - sampler 설정 
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, opts.batch_size, drop_last=True)

    val_sampler = DistributedSampler(dataset=val_dataset, shuffle=False)
    batch_sampler_val = torch.utils.data.BatchSampler(val_sampler, opts.batch_size, drop_last=False)

    train_dataloader = data.DataLoader(
        train_dataset, batch_sampler=batch_sampler_train, num_workers=opts.num_workers, collate_fn=od_collate_fn)
    
    val_dataloader = data.DataLoader(
        val_dataset, batch_sampler=batch_sampler_val, num_workers=opts.num_workers,collate_fn=od_collate_fn)
    
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


    # model 감싸기 
    net.cuda(local_gpu_id)
    net = DDP(module=net, device_ids=[local_gpu_id])


    # 4. 손실함수 설정
    criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=local_gpu_id)

    # 5. 최적화 기법 설정
    optimizer = optim.SGD(net.parameters(), lr=opts.lr, momentum=0.9, weight_decay=5e-4)

    # 6. 학습 및 검증 실시
    num_epochs = 50

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
                train_sampler.set_epoch(epoch)  # 에폭마다 데이터셋을 새로 섞는다
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
            for images, targets in dataloaders_dict[phase]:

                #GPU를 사용할 수 있으면 GPU에 데이터를 보낸다
                images = images.to(local_gpu_id)
                targets = [ann.to(local_gpu_id) for ann in targets]   # 리스트 각 요소의 텐서를 GPU로

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

                        if(iteration % 10 == 0):    #10iter에 한번 검증한다
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
            if phase == "train":
                wandb.log({"train_loss": epoch_train_loss})
            elif (phase == "val") and ((epoch+1) % 10 == 0):
                wandb.log({"val_loss": epoch_val_loss})
            

            epoch_train_loss = 0.0  # epoch의 손실합
            epoch_val_loss = 0.0  # epoch의 손실합

            # 네트워크를 저장한다
            if(((epoch+1) % 10 == 0) and (opts.rank == 0)):
                torch.save(net.state_dict(), 'weights/ssd300_' + str(epoch+1) + '.pth')
                #dist.barrier()

if __name__ == '__main__':
    
    new_directory = './ssd0/byeongcheol/SSD-Practice/'
    os.chdir(new_directory)

    # parser 설정
    parser = argparse.ArgumentParser('Distributed training test', parents=[get_args_parser()])
    opts = parser.parse_args()
    opts.ngpus_per_node = torch.cuda.device_count()
    opts.gpu_ids = list(range(opts.ngpus_per_node))
    opts.num_workers = opts.ngpus_per_node * 4


 

    mp.spawn(main, args=(opts,), nprocs=opts.ngpus_per_node, join=True)
