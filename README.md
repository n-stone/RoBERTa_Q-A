# RoBERTa_Fine Tuning

## 1. Introdue
1-1. RoBERTa Fine-Tuning Model
- [**RoBERTa**](https://huggingface.co/klue/roberta-base) 모델 기반의 Question & Answer 대화 시스템


1-2. DATA SET
- [**AI HUB**](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86) 데이터를 이용하였습니다.


## 2. USED
2-1. Install 


>pip install -r requirements.txt
>$ RoBERTa.yaml


2-2. Train 
- conifg.py는 환경 변수 정의
> python train.py


2-3. Test

- 채팅 사용

> python print_result.py --chat CHAT   

- 결과 사용

> python print_result.py --test TEST   