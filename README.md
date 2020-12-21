# 한국어(NSMC)/영어(Friends) KoELECTRA/ELECTRA를_이용한_감정분석기 (Pytorch + HuggingFace)
# Colab 에서 개발 및 실행 (하단 실행방법 참조)
# Github : https://github.com/MunJinSeo/MyProject/
<br>

## References 1
- (1) 김희규님의 "HuggingFace KoElectra로 NSMC 감성분석 Fine-tuning해보기"<br>
https://heegyukim.medium.com/huggingface-koelectra%EB%A1%9C-nsmc-%EA%B0%90%EC%84%B1%EB%B6%84%EB%A5%98%EB%AA%A8%EB%8D%B8%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B0-1a23a0c704af

- (2) 이지원님의 Github : nlp_emotion_classification <br>
https://github.com/jiwonny/nlp_emotion_classification

## 사용모델: 한국어 KoELECTRA , 영어 ELECTRA
- 한국어 : 박장원님의 KoELECTRA-base-v3 사용<br>
https://monologg.kr/2020/05/02/koelectra-part1/<br>
https://github.com/monologg/KoELECTRA
- 영어 : 구글 ELECTRA-large 사용<br>
https://github.com/google-research/electra <br>
https://huggingface.co/google/electra-large-discriminator<br>
참고 : https://huggingface.co/google/electra-base-discriminator<br>
참고 : https://huggingface.co/google/electra-small-discriminator<br>

## Dataset (학습에는 train파일만 사용함, test파일은 검증에 사용)
- 한국어 : 네이버 영화 리뷰 데이터셋<br>
https://github.com/e9t/nsmc <br>
ratings_train.txt <br>
ratings_test.txt
- 영어 : Freinds <br>
http://doraemon.iis.sinica.edu.tw/emotionlines/ <br>
friends_train.json <br>
friends_test.json <br>
friends_dev.json

## 과제 파일
- 한국어 : https://www.kaggle.com/c/korean-sa-competition-bdc101 <br>
ko_data.csv
- 영어 : https://www.kaggle.com/c/english-sa-competition-bdc101 <br>
en_data.csv

## References 2
- https://colab.research.google.com/drive/1tIf0Ugdqg4qT7gcxia3tL7und64Rv1dP
- https://blog.naver.com/horajjan/221739630055
<br>@전처리 관련@<br>
- https://github.com/YongWookHa/kor-text-preprocess
- https://github.com/likejazz/korean-sentence-splitter
- https://github.com/lovit/soynlp
<br>@@<br>
- https://huggingface.co/transformers/training.html
- https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
- https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html
- https://wikidocs.net/44249


## 실행 방법 / 유의사항
- Colab 에서 반드시 GPU로 실행 (Colab Pro 권장)
- 과제파일은 사전에 서버로 업로드
- 모델 학습을 위한 데이터셋은 사전에 서버로 업로드<br>
  단, 한국어는 자동 다운로드 처리가능, 영어는 수동 업로드
- 소스는 위에서부터 순차적으로 실행하면 됨
- CUDA(GPU) 메모리 오버되는경우 학습시 Batch size 줄여서 해볼것
- KoELECTRA base_v3으로 사용했으며 1epoch 당 약 60분
- ELECTRA large 사용했으며 1epoch 당 약 30분


## 처리 순서 
- 필요 lib 설치
- NSMC / Friends 데이터셋 처리
- 필요 모듈 import
- 데이터셋 처리 (Dataset Calss / 전처리)
- 모델 생성 (Create Model)
- 학습(Learn) - train파일만 사용(test제외)
- 테스트 데이터셋 정확도 확인하기
- 모델 저장하기
- 과제용 데이터 예측 및 맵핑
- 결과 파일 저장



