# 3D FRONT Integration for LP2

이 README는 3D FRONT 데이터셋을 LP2에 적용하는 빠른 시작 가이드입니다.

## 빠른 시작

### 1. 필수 조건

- Python 3.10+
- LP2 설치 완료 (`pip install -r requirements.txt`)
- OpenAI API 키 설정 (`.env` 또는 환경 변수)
- 3D FRONT scene JSON 파일

### 2. 한 번에 실행하기

```bash
# Example workflow script 사용
bash scripts/example_3dfront_workflow.sh \
    path/to/your/3dfront_scene.json \
    my_scene_001
```

이 스크립트는 다음을 자동으로 수행합니다:
1. 3D FRONT → Spark-DSG 변환
2. Synthetic trajectories 생성
3. Scene bounds 계산
4. Config 파일 생성

### 3. LP2 실행

```bash
python src/lhmp/main.py --run_pipeline "y" \
    --global_config_file "config/global_config.yaml" \
    --method_config_file "config/method_configs/project_config_LP2.yaml" \
    --scene_config_file "config/scene_configs/scene_config_3dfront_my_scene_001.yaml"
```

## 단계별 실행

### Step 1: Scene 변환

```bash
python scripts/convert_3dfront_to_dsg.py \
    --input_json path/to/3dfront_scene.json \
    --output_dir data/3dfront/scene_001/ \
    --scene_id scene_001
```

**출력:**
- `scene_001_scene_graph.json` - Spark-DSG 형식
- `scene_001_room_labels.json` - Room labels

### Step 2: Trajectory 생성

#### Option A: Synthetic Trajectories

```bash
python scripts/generate_synthetic_trajectories.py \
    --scene_graph data/3dfront/scene_001/scene_001_scene_graph.json \
    --room_labels data/3dfront/scene_001/scene_001_room_labels.json \
    --output_dir data/3dfront/scene_001/trajectories/ \
    --num_trajectories 10 \
    --duration 180.0
```

#### Option B: 기존 Trajectory 변환

```bash
python scripts/convert_trajectory_to_lp2.py \
    --input_trajectory path/to/trajectory.txt \
    --input_format xyz \
    --scene_graph data/3dfront/scene_001/scene_001_scene_graph.json \
    --output_dir data/3dfront/scene_001/trajectories/ \
    --trajectory_id traj_001
```

### Step 3: Scene Bounds 계산

```bash
python scripts/compute_scene_bounds.py \
    --scene_graph data/3dfront/scene_001/scene_001_scene_graph.json \
    --padding 1.0
```

출력된 `environment_bounds`를 복사하여 config 파일에 사용합니다.

### Step 4: Config 파일 생성

`config/scene_configs/scene_config_3dfront_template.yaml`을 복사하고 수정:

```yaml
data:
  scene_graph_path: data/3dfront/scene_001/scene_001_scene_graph.json
  room_labels_path: data/3dfront/scene_001/scene_001_room_labels.json
  trajectory_dir: data/3dfront/scene_001/trajectories

global:
  scene: 3dfront_scene_001
  environment_bounds: [[-5.2, -4.1], [8.3, 6.7]]  # From compute_scene_bounds
  bandwidth: 0.8
```

## 생성된 파일들

```
LP2/
├── scripts/
│   ├── convert_3dfront_to_dsg.py       # 3D FRONT → DSG 변환
│   ├── generate_synthetic_trajectories.py  # Synthetic trajectory 생성
│   ├── convert_trajectory_to_lp2.py    # Trajectory 형식 변환
│   ├── compute_scene_bounds.py         # Scene bounds 계산
│   └── example_3dfront_workflow.sh     # 전체 워크플로우 예제
├── docs/
│   └── 3DFRONT_INTEGRATION.md          # 상세 문서
└── config/
    └── scene_configs/
        └── scene_config_3dfront_template.yaml  # Config 템플릿
```

## 주요 스크립트 옵션

### convert_3dfront_to_dsg.py

```bash
--input_json        # 3D FRONT JSON 파일 경로 (필수)
--output_dir        # 출력 디렉토리 (필수)
--scene_id          # Scene 식별자 (기본값: 3dfront_scene)
```

### generate_synthetic_trajectories.py

```bash
--scene_graph       # Scene graph 경로 (필수)
--room_labels       # Room labels 경로 (필수)
--output_dir        # 출력 디렉토리 (필수)
--num_trajectories  # 생성할 trajectory 개수 (기본값: 10)
--duration          # Trajectory 길이 (초) (기본값: 180.0)
--min_interactions  # 최소 상호작용 수 (기본값: 3)
--max_interactions  # 최대 상호작용 수 (기본값: 7)
--walking_speed     # 걷기 속도 (m/s) (기본값: 1.0)
--seed              # Random seed (재현성)
```

### convert_trajectory_to_lp2.py

```bash
--input_trajectory  # 입력 trajectory 파일 (필수)
--input_format      # 입력 형식: xyz, json, csv (기본값: xyz)
--scene_graph       # Scene graph 경로 (필수)
--output_dir        # 출력 디렉토리 (필수)
--trajectory_id     # Trajectory ID (기본값: traj)
--interaction_threshold     # 상호작용 거리 임계값 (m) (기본값: 0.5)
--min_interaction_duration  # 최소 상호작용 시간 (s) (기본값: 3.0)
```

### compute_scene_bounds.py

```bash
--scene_graph       # Scene graph 경로 (필수)
--padding           # Bounds 주변 여백 (m) (기본값: 1.0)
```

## 트러블슈팅

### 1. "No module named 'spark_dsg'"

```bash
pip install spark-dsg
```

### 2. 3D FRONT JSON 구조가 다른 경우

`scripts/convert_3dfront_to_dsg.py`의 140번째 줄 근처에서 필드 이름 수정:

```python
scene_info = scene_data.get("scene", scene_data)  # 또는 다른 키
rooms = scene_info.get("room", [])                # 또는 다른 키
furniture_list = scene_info.get("furniture", [])  # 또는 다른 키
```

### 3. Trajectory가 너무 짧음

`generate_synthetic_trajectories.py`에서 duration 증가:

```bash
--duration 300.0  # 5분
--max_interactions 10
```

### 4. Room/Furniture 카테고리 매핑 추가

`convert_3dfront_to_dsg.py`에서 매핑 딕셔너리 수정:

```python
self.room_type_mapping = {
    # 기존 매핑...
    "YourCustomRoomType": "custom room label",
}

self.furniture_category_mapping = {
    # 기존 매핑...
    "your_furniture": "custom furniture label",
}
```

## 예제: 완전한 워크플로우

```bash
# 1. Scene 변환
python scripts/convert_3dfront_to_dsg.py \
    --input_json ~/3DFRONT/living_room.json \
    --output_dir data/3dfront/living_room/ \
    --scene_id living_room_001

# 2. Trajectory 생성
python scripts/generate_synthetic_trajectories.py \
    --scene_graph data/3dfront/living_room/living_room_001_scene_graph.json \
    --room_labels data/3dfront/living_room/living_room_001_room_labels.json \
    --output_dir data/3dfront/living_room/trajectories/ \
    --num_trajectories 20 \
    --duration 180.0 \
    --seed 42

# 3. Bounds 계산
python scripts/compute_scene_bounds.py \
    --scene_graph data/3dfront/living_room/living_room_001_scene_graph.json

# 4. Config 생성 (bounds를 위 출력에서 복사)
cat > config/scene_configs/scene_config_3dfront_living_room.yaml << 'EOF'
data:
  scene_graph_path: data/3dfront/living_room/living_room_001_scene_graph.json
  room_labels_path: data/3dfront/living_room/living_room_001_room_labels.json
  trajectory_dir: data/3dfront/living_room/trajectories

global:
  scene: 3dfront_living_room_001
  environment_bounds: [[-6.5, -5.2], [9.3, 7.8]]
  bandwidth: 0.8
EOF

# 5. LP2 실행
python src/lhmp/main.py --run_pipeline "y" \
    --global_config_file "config/global_config.yaml" \
    --method_config_file "config/method_configs/project_config_LP2.yaml" \
    --scene_config_file "config/scene_configs/scene_config_3dfront_living_room.yaml"

# 6. 결과 시각화
python scripts/visualize_all.py \
    --methods "LP2" \
    --scenes "3dfront_living_room_001" \
    --n_past_interactions 2 \
    --time_lim 60.0
```

## 더 많은 정보

- 상세 문서: [docs/3DFRONT_INTEGRATION.md](docs/3DFRONT_INTEGRATION.md)
- LP2 원본 README: [README.md](README.md)
- 3D FRONT 데이터셋: https://tianchi.aliyun.com/dataset/65347
- Spark-DSG: https://github.com/MIT-SPARK/Spark-DSG

## 문의

Issues 및 Pull Requests는 GitHub 리포지토리에 제출해주세요.
