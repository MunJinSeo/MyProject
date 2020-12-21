# 한국어(NSMC)/영어(Friends) KoELECTRA/ELECTRA를_이용한_감정분석기_학습 (Pytorch + HuggingFace)
# Colab 에서 개발 및 실행
# Github : https://github.com/MunJinSeo/MyProject/
<br>

## References 1
- 김희규님의 "HuggingFace KoElectra로 NSMC 감성분석 Fine-tuning해보기"<br>
https://heegyukim.medium.com/huggingface-koelectra%EB%A1%9C-nsmc-%EA%B0%90%EC%84%B1%EB%B6%84%EB%A5%98%EB%AA%A8%EB%8D%B8%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B0-1a23a0c704af

- 이지원님의 Github : nlp_emotion_classification <br>
https://github.com/jiwonny/nlp_emotion_classification

## 사용모델 KoELECTRA , ELECTRA
- 한국어 : 박장원님의 KoELECTRA 사용<br>
https://monologg.kr/2020/05/02/koelectra-part1/<br>
https://github.com/monologg/KoELECTRA
- 영어 : 구글 ELECTRA 사용<br>
https://huggingface.co/google/electra-small-discriminator<br>
https://github.com/google-research/electra

## Dataset (학습에는 train파일만 사용함, test파일 사용시 감점대상)
- 한국어 : 네이버 영화 리뷰 데이터셋<br>
https://github.com/e9t/nsmc
- 영어 : Freinds <br>
http://doraemon.iis.sinica.edu.tw/emotionlines/

## References 2
- https://colab.research.google.com/drive/1tIf0Ugdqg4qT7gcxia3tL7und64Rv1dP
- https://blog.naver.com/horajjan/221739630055 <br>@전처리 관련@<br>
- https://github.com/YongWookHa/kor-text-preprocess
- https://github.com/likejazz/korean-sentence-splitter
- https://github.com/lovit/soynlp <br>@@<br>
- https://huggingface.co/transformers/training.html
- https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
- https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html
- https://wikidocs.net/44249


## 기타/유의사항
- 반드시 GPU로 실행 (Colab Pro 권장)
- CUDA(GPU) 메모리 오버되는경우 학습시 Batch size 줄여서 해볼것
- 학습용 데이터 파일은 사전에 업로드 할것
- ELECTRA large 1epoch 당 약 30분
- KoELECTRA base_v3 1epoch 당 약 60분

