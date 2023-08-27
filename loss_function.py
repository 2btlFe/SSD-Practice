# 패키지 import
import torch
import torch.nn as nn
import torch.nn.functional as F

# "utils" 폴더에 있는 match 함수를 기술한 match.py를 import
from utils.match import match

class MultiBoxLoss(nn.Module):
    """SSD의 손실함수 클래스입니다"""

    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh  # 0.5 match 함수의 jaccard 계수의 임계치
        self.negpos_ratio = neg_pos  # 3:1 Hard Negative Mining의 음과 양의 비율
        self.device = device  # CPU와 GPU 중 어느 것으로 계산하는가

    def forward(self, predictions, targets):
        """
        손실 함수 계산

        Parameters
        ----------
        predictions : SSD net의 훈련시의 출력(tuple)
            (loc=torch.Size([num_batch, 8732, 4]), conf=torch.Size([num_batch, 8732, 21]), dbox_list=torch.Size [8732,4])

        targets : [num_batch, num_objs, 5]
            5는 정답의 어노테이션 정보[xmin, ymin, xmax, ymax, label_ind]를 나타낸다

        Returns
        -------
        loss_l : 텐서
            loc의 손실값
        loss_c : 텐서
            conf의 손실값
        """
        # SSD 모델의 출력이 튜플로 되어 있으므로 개별적으로 해체한다
        loc_data, conf_data, dbox_list = predictions
        # loc의 크기는 torch.Size([batch_num, 8732, 4])
        # conf의 크기는 torch.Size([batch_num, 8732, 21])
        #DBox를 텐서로 변환 .torch.Size([8732, 4])

        # 요소 수 파악
        num_batch = loc_data.size(0)    # 미니 배치 크기
        num_dbox = loc_data.size(1)     # DBox 수 = 8732
        num_classes = conf_data.size(2) # 클래스 수 = 21

        # 손실 계산에 사용할 것을 저장하는 변수를 작성
        # conf_t_label: 각 DBox에 가장 가까운 정답 BBox의 라벨을 저장   [batch, 8732]
        # loc_t: 각 DBox에 가장 가까운 정답 BBox의 위치 정보를 저장     [batch, 8732, 4]
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)    
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)            

        # loc_t와 conf_t_label에,
        # DBox와 정답 어노테이션 targets를 match 시킨 결과를 덮어쓰기
        for idx in range(num_batch):    # 미니배치 루프
            
            # 현재 미니 뱃지의 정답 어노테이션의 BBox와 라벨을 취득
            truths = targets[idx][:, :-1].to(self.device)   #BBox - label index만 제외
            # 라벨 [물체1의 라벨, 물체2의 라벨, …]
            labels = targets[idx][:, -1].to(self.device)

            # 디폴트 박스를 새로운 변수로 준비
            dbox = dbox_list.to(self.device)

            # match 함수를 실행하여 loc_t와 conf_t_label의 내용을 갱신한다
            # (상세)
            # loc_t: 각 DBox에 가장 가까운 정답 BBox의 위치 정보가 덮어써진다
            # conf_t_label: 각 DBox에 가장 가까운 BBox의 라벨이 덮어써진다
            # 단, 가장 가까운 BBox와의 jaccard overlap이 0.5보다 작은 경우
            # 정답 BBox의 라벨 conf_t_label은 배경 클래스 0으로 한다
            variance = [0.1, 0.2]
            # 이 variance는 DBox에서 BBox로 보정 계산할 때 사용하는 식의 계수입니다
            match(self.jaccard_thresh, truths, dbox, variance, labels, loc_t, conf_t_label, idx)

        #-------------
        # 위치의 손실: loss_l을 계산
        # Smooth L1 함수로 손실을 계산한다. 단, 물체를 발견한 DBox의 오프셋만 계산한다
        #-------------
        # 물체를 감지한 BBox를 꺼내는 마스크를 작성
        pos_mask = conf_t_label > 0 # torch.Size([num_batch, 8732])

        # pos_mask를 loc_data 크기로 변형 - [num_batch, 8732, 4]
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # Positive DBox의 loc_data와 지도 데이터 loc_t를 취득
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        # 물체를 발견한 Positive DBox의 오프셋 정보 loc_t의 손실(오차)을 계산
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        #-------------
        # 클래스 예측의 손실: loss_c로 계산
        # 교차 엔트로피 오차 함수로 손실 계산. 단, 배경 클래스가 정답인 DBox가 압도적으로 많으므로
        # Hard Negative Mining을 실시하여 물체 발견 DBox 및 배경 클래스 DBox의 비율이 1:3이 되도록 한다.
        # 배경 클래스 DBox로 예상한 것 중 손실이 적은 것은 클래스 예측 손실에서 제외
        #-------------
        batch_conf = conf_data.view(-1, num_classes)
        # batch_conf([num_batch * 8732, 21])

        # 클래스 예측의 손실함수 계산(reductino = 'none'으로 하여 합을 취하지 않고 차원을 보존)
        loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction='none')
        

        #------------
        # Negative DBox 중 Hard Negative Mining으로 추출하는 것을 구하는 마스크 작성
        #------------

        # 물체를 발견한 Positive DBox의 손실을 0으로 한다.
        # 물체 발견한 Positive DBox의 손실을 0으로 한다
        # (주의)물체는 label이 1 이상으로 되어 있다. 라벨 0은 배경을 의미.
        num_pos = pos_mask.long().sum(1, keepdim=True) # 미니배치별 물체 클래스 예측의 수
        loss_c = loss_c.view(num_batch, -1) #torch.Size([num_batch, 8732])
        loss_c[pos_mask] = 0    #물체를 발견한 DBox는 손실 0으로 한다

        # Hard Negative Mining을 실시
        # 각 DBox 손실의 크기 loss_c의 순위 idx_rank을 구한다
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        # (주석)
        # 구현된 코드는 특수하며 직관적이지 않습니다.
        # 위 두 줄의 요점은 각 DBox에 대해, 손실 크기가 몇 번째인지의 정보를
        # 변수 idx_rank에 고속으로 취득하는 코드입니다.
        
        # DBox의 손실 값이 큰 쪽에서 내림차순으로 정렬하여, DBox의 내림차순의 index를 loss_idx에 저장한다.
        # 손실 크기 loss_c의 순위 idx_rank를 구한다.
        # 여기서 내림차순이 된 배열 index인 loss_idx를 0부터 8732까지 오름차순으로 다시 정렬하기 위해서는,
        # 몇 번째 loss_idx의 인덱스를 취할지를 나타내는 것이 idx_rank이다.
        # 예를 들면, idx_rank 요소의 0번째 = idx_rank[0]를 구하려면 loss_idx의 값이 0인 요소,
        # 즉 loss_idx[?]=0의, ?는 몇 번째를 구할 것인지가 된다. 여기서 ? = idx_rank[0] 이다.
        # 지금 loss_idx[?]=0의 0은, 원래 loss_cの의 요소 0번째라는 의미이다.
        # 즉 ?은 원래 loss_c의 요소 0번째는, 내림차순으로 정렬된 loss_idx의 몇 번째입니까?
        # 를 구하는 것이 되어, 결과적으로
        # ? = idx_rank[0] 은 loss_c의 요소 0번째가 내림차순으로 몇 번째인지 나타낸다.

        # 배경 DBox의 수 num_neg를 구한다. HardNegative Mining에 의해,
        # 물체 발견 DBox의 수 num_pos의 3배(self.negpos_ratio 배)로 한다
        # 만에 하나, DBox의 수를 초과한 경우에는 DBox의 수를 상한으로 한다
        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)

        # idx_rank는 각 DBox의 손실의 크기가 위에서 부터 몇 번째인지가 저장되어 있다
        # 배경 DBox의 수 num_neg보다 순위가 낮은(손실이 큰) DBox를 취하는 마스크를 작성
        # torch.Size([num_batch, 8732])
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # -----------------
        # (종료) 이제부터 Negative DBox 중에서, Hard Negative Mining로 추출할 것을 구하는 마스크를 작성합니다
        # -----------------

        # 마스크의 모양을 성형하여 conf_data에 맞춘다
        # pos_idx_mask는 Positive DBox의 conf를 꺼내는 마스크입니다
        # neg_idx_mask는 Hard Negative Mining으로 추출한 Negative DBox의 conf를 꺼내는 마스크입니다
        # pos_mask: torch.Size([num_batch, 8732])→pos_idx_mask: torch.Size([num_batch, 8732, 21])
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # conf_data에서 pos와 neg만 꺼내서 conf_hnm으로 한다. 형태는 torch.Size([num_pos+num_neg, 21])
        conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)
                             ].view(-1, num_classes)
        # (주석) gt는 greater than(>)의 약칭. mask가 1인 index를 꺼낸다
        # pos_idx_mask+neg_idx_mask는 덧셈이지만, index로 mask를 정리할 뿐이다
        # 즉, pos이든 neg이든, 마스크가 1인 것을 더해 하나의 리스트로 만들어, 이를 gt로 취득

        # 마찬가지로 지도 데이터인 conf_t_label에서 pos와 neg만 꺼내어 conf_t_label_hnm으로
        # torch.Size([pos+neg]) 형태가 된다
        conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]

        # confidence의 손실함수를 계산(요소의 합계=sum을 구한다)
        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

        # 물체를 발견한 BBox의 수 N(전체 미니 배치의 합계)으로 손실을 나눔
        N = num_pos.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c

        

