import cv2
import matplotlib.pyplot as plt 
import numpy as np
import torch
from ssd_model_forwad import SSD
from Dataset_DataLoader import DataTransform
from utils.ssd_predict_show import SSDPredictShow



if __name__ == "__main__":

    
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']

    # SSD300 설정
    ssd_cfg = {
        'num_classes': 21,  # 배경 클래스를 포함한 총 클래스 수
        'input_size': 300,  # 화상의 입력 크기
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 출력할 DBox 화면비 종류
        'feature_maps': [38, 19, 10, 5, 3, 1],  # 각 source의 화상 크기
        'steps': [8, 16, 32, 64, 100, 300],  # DBOX의 크기를 결정
        'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOX의 크기를 결정
        'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOX의 크기를 결정
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

    # SSD 네트워크 모델
    net = SSD(phase="inference", cfg=ssd_cfg)

    # SSD의 학습된 가중치 설정
    net_weights = torch.load('./weights/ssd300_10.pth', map_location={'cuda:0': 'cpu'})

    #net_weights = torch.load('./weights/ssd300_mAP_77.43_v2.pth', map_location={'cuda:0': 'cpu'})


    net.load_state_dict(net_weights)

    print('네트워크 설정 완료: 학습된 가중치를 로드했습니다')

    # 1. 화상 읽기 
    image_file_path = "./data/cowboy-757575_640.jpg"
    img = cv2.imread(image_file_path)   #[높이][폭][색BGR]
    height, width, channels = img.shape #화상 크기 취득

    # 2. 원본 화상 표시
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    #BGR -> RGB로 변경
    plt.show()

    # 3. 전처리 클래스 작성
    color_mean = (104, 117, 123)  # (BGR)의 색의 평균값
    input_size = 300  # 화상의 input 크기를 300×300으로 설정
    transform = DataTransform(input_size, color_mean)

    # 4. 전처리
    phase = "val"
    img_transformed, boxes, labels = transform(
        img, phase, "", "") #어노테이션은 없어 ""로 설정
    img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

    # 5. SSD로 예측
    net.eval()  # 네트워크를 추론 모드로 
    x = img.unsqueeze(0)    # 미니 배치화: torch.Size([1, 3, 300, 300])
    detections = net(x)

    print(detections.shape)
    print(detections)
        
    # output : torch.Size([batch_num, 21, 200, 5])
    #  = (batch_num, 클래스, conf의 top200, 규격화된 BBox의 정보)
    #   규격화된 BBox의 정보(신뢰도, xmin, ymin, xmax, ymax)

    # 파일 경로
    ssd = SSDPredictShow(eval_categories=voc_classes, net=net)
    ssd.show(image_file_path, data_confidence_level=0.6)
    print("show 성공")















