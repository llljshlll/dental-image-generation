# ctrLoRA training

### 1. 데이터 준비
lighting map과 segmentation map을 blending하여 데이터셋 구성
<img src="images/input_blending.png" alt="lighting map, segmentation map blending" width="300">
샘플 수 : 2000
source : blending image
target : target image
prompt : 치아 png 파일 속 제목에 따라 ""로 설정

### 2. 2. forward Process
1. 원본이미지 x_0하나를 가져옴
2. 거기서 랜덤 타임스텝t를 하나 뽑음 ex) 327/1000
3. 그 t에 대항하는 노이즈 비율(a_t)로 노이즈를 추가해서 (x_t=그 식)이라는 노이즈 낀 이미지 만듦
<img src="images/forward_noise_add_process.png" alt="forward process" width="300">


5000스텝 
배치사이즈 1
RTX 4090기준 2시간 30분

