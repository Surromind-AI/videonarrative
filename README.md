# videonarrative# videonarrative
## 1. test data 선 feature 추출, 후 추론 실행

1. 소스코드 메인 디렉토리인 "/data/dekim/video-qa/video-qa/hcrn-videoqa-master"로 디렉토리 이동

```bash
cd /data/dekim/video-qa/video-qa/hcrn-videoqa-master
```

1. language feature 추출 진행
    
    1) 학습 language feature 추출 명령어
    
    ```bash
    python3 preprocess/preprocess_questions.py --dataset video-narr --glove_pt /home/word-embeddings/glove/glove.korean.pkl --mode train --video_dir {비디오경로}
    ```
    
2. video feature 추출 진행
    
    1) video appearance feature 추출 명령어
    
    ```bash
    python3 preprocess/preprocess_features.py --gpu_id 0 --dataset video-narr --model resnet101 --video_dir {비디오경로}
    ```
    
    2) video motion feature 추출 명령어
    
    ```bash
    python3 preprocess/preprocess_features.py --gpu_id 0 --dataset video-narr --model resnext101 --image_height 112 --image_width 112 --video_dir {video 경로}
    ```
    
3. 데이터 검증 위한 추론 진행

```bash
python3 python3 validate.py --cfg configs/video_narr.yml
```
