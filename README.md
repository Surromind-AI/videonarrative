# 2021 한국어 질의응답 AI 경진대회 - Baseline 코드 안내
## 환경설정

<aside>
❗ 1. 원천 데이터(raw data)와 라벨링 데이터의 구조는 [데이터 구조]와 같다고 가정한다.
2. 모든 명령은 baseline_code 폴더 아래에서 실행한다.

</aside>

## [데이터 구조]

```bash
{사용자 지정 경로}/원천데이터
								ㄴ예능교양
									ㄴ 대본X
									ㄴ 대본O
								ㄴ스포츠
									ㄴ 대본X
								ㄴ생활안전
									ㄴ 대본X
									ㄴ 대본O

{사용자 지정 경로}/라벨링데이터
								ㄴ예능교양
									ㄴ 대본X
									ㄴ 대본O
								ㄴ스포츠
									ㄴ 대본X
								ㄴ생활안전
									ㄴ 대본X
									ㄴ 대본O
```

## 공통

** {사용자 지정 data경로} 는 data의 '원천데이터' 혹은 '라벨링데이터' 상위경로까지만 작성한다.

ex. data 위치 경로가 "/home/data/원천데이터 혹은 라벨링데이터"일 경우

명령어는 아래와 같다.

```bash
# 테스트 language-feature 추출
python3 preprocess/preprocess_questions.py --dataset video-narr --glove_pt /home/word-embeddings/glove/glove.korean.pkl --mode test --video_dir /home/data
```

## 0. 사전 준비사항

### Raw data 혹은 feature 파일 다운로드

아래 링크에서 다운

```bash
#링크 적어주세요
```

### glove.korean.pkl 파일 다운로드

아래 링크에서 다운로드

```bash
https://drive.google.com/file/d/1GQtrtgX8WsTO9mjo61qjj5xCozcvAmxe/view?usp=sharing
```

## 1. feature 추출 진행

1) 학습 language feature 추출 명령어

```bash
python3 preprocess/preprocess_questions.py --dataset video-narr --glove_pt /home/word-embeddings/glove/glove.korean.pkl --mode train --video_dir {비디오경로}
```

1. video feature 추출 진행
    
    1) video appearance feature 추출 명령어
    
    ```bash
    python3 preprocess/preprocess_features.py --gpu_id 0 --dataset video-narr --model resnet101 --video_dir {비디오경로}
    ```
    
    2) video motion feature 추출 명령어
    
    ```bash
    python3 preprocess/preprocess_features.py --gpu_id 0 --dataset video-narr --model resnext101 --image_height 112 --image_width 112 --video_dir {video 경로}
    ```
    

## 2. 학습 진행

```bash
	python3 train.py --cfg configs/video_narr.yml
```

## 3. 검증 진행

```bash
python3 validate.py --cfg configs/video_narr.yml
```
