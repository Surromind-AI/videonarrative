# 2021 한국어 질의응답 AI 경진대회 - Baseline 코드 안내
## 환경설정

<aside>
❗ 1. 원천데이터와 라벨링 데이터 구조는 [데이터 구조]와 같다고 가정한다.
	
2. 모든 명령은 baseline_code 폴더 아래에서 실행한다.

</aside>

## [데이터 구조]

### Train 데이터
```bash
{사용자 지정 경로}/raw_data/train/원천데이터
	ㄴ예능교양
		ㄴ 대본X
		ㄴ 대본O
	ㄴ스포츠
		ㄴ 대본X
	ㄴ생활안전
		ㄴ 대본X
		ㄴ 대본O

{사용자 지정 경로}/raw_data/train/라벨링데이터
	ㄴ예능교양
		ㄴ 대본X
		ㄴ 대본O
	ㄴ스포츠
		ㄴ 대본X
	ㄴ생활안전
		ㄴ 대본X
		ㄴ 대본O
```
### Test 데이터
```bash
{사용자 지정 경로}/raw_data/test/원천데이터
	ㄴ예능교양
		ㄴ 대본X
		ㄴ 대본O
	ㄴ스포츠
		ㄴ 대본X
	ㄴ생활안전
		ㄴ 대본X
		ㄴ 대본O

{사용자 지정 경로}/raw_data/test/라벨링데이터
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
python3 preprocess/preprocess_questions.py --dataset video-narr --glove_pt /{사용자 지정 경로}/word-embeddings/glove/glove.korean.pkl --mode test --video_dir /home/data
```

## 0. 사전 준비사항

### Raw data 혹은 feature 파일 다운로드

아래 클릭하여 다운로드

- [raw data](https://drive.google.com/file/d/1fbMB1XQvJCa2ODV0ssHYSlSXGf1fe13E/view?usp=sharing)
- [feature data](https://drive.google.com/file/d/15dUXKfrR5eUAa2NIK_Oid6SdTcvyfzPg/view?usp=sharing)

### glove.korean.pkl 파일 다운로드

아래 클릭하여 다운로드

- [glove.korean.pkl](https://drive.google.com/file/d/1rl3BiT7OcOzakPEp3JGmeEZBSAaM2B4Y/view?usp=sharing)

### renet, renext binary 파일 다운로드

```bash
#링크 적어주세요
```

### 리눅스 한글 세팅

[출처] : [https://epicarts.tistory.com/30](https://epicarts.tistory.com/30) [일상 생활]

- 1 - 한글 언어팩 설치
    
    ```bash
    apt-get isntall language-pack-ko
    ```
    
- 2 - locale 생성
    
    ```bash
    locale gen ko.KR.UTF-8
    ```
    
- 3 - locale 설정
    - 3-1 locale 파일 open
        
        ```bash
        vim /etc/default/locale
        ```
        
    - 3-2 LANG option 추가 입력 (en_US.UTF-8 의 경우 입력되어 있으면 입력할 필요 없음)
        
        ```bash
        LANG=en_US.UTF-8
        LANG=ko_KR.UTF-8
        ```
        
- 4 - environment 설정
    - 4-1 environment 파일 open
        
        ```bash
        vim /etc/environment
        ```
        
    - 4-2 LANG option 추가 입력
        
        ```bash
        LANG=ko_KR.UTF8
        LANGUAGE=ko_KR:ko:en_GB:en
        ```
        
- 5 - 폰트 설치
    
    ```bash
    apt-get install fonts-nanum*
    ```
    
- 6 - 재부팅
    
    ```bash
    reboot / init 6
    ```
    

## 한국어 NLP를 위한 형태소 분석기 Mecab 설치

[출처] : [https://i-am-eden.tistory.com/9](https://i-am-eden.tistory.com/9)

- 1 - JDK 설치
    
    ```bash
    sudo apt-get install openjdk-8-jdk python-dev 
    sudo apt-get install python3-dev
    ```
    
- 2 - KoNLPy 설치
    
    ```bash
    pip3 install konlpy
    ```
    
- 3 - Mecab 설치
    
    ```bash
    wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
    tar -zxvf mecab-*-ko-*.tar.gz
    
    cd mecab-0.996-ko-0.9.2/
    ./configure
    make
    make check
    sudo make install
    ```
    
- 4 - Mecab-ko-dic 사전 설치
    
    ```bash
    wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.0.1-20150920.tar.gz
    tar -zxvf mecab-ko-dic-2.0.1-20150920.tar.gz
    cd mecab-ko-dic-2.0.1-20150920/
    ./autogen.sh
    ./configure
    make
    sudo make install
    cd ..
    ```
    
    ```bash
    git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git
    cd mecab-python-0.996/
    python3 setup.py build
    python3 setup.py install
    cd ..
    pwd
    ```
    

## 필요한 라이브러리 설치

### requirements.txt 설치

```bash
../baseline_code$ pip3 install -r requirements.txt
```

### ffmpeg 설치

```bash
pip3 install ffmpeg
```

## 1. feature 추출 진행

1) 학습 language feature 추출 명령어

```bash
python3 preprocess/preprocess_questions.py --dataset video-narr --glove_pt /{사용자 지정 경로}/word-embeddings/glove/glove.korean.pkl --mode train --video_dir {비디오경로}
```

<aside>
📌 해당 language feature 추출 진행 시,
학습 데이터셋의 10%는 검증 데이터로 사용된다.

e.g) 학습 데이터셋 : 100개
실제 학습 데이터셋 : 90개
검증 데이터 : 10개

</aside>

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
