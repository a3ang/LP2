# 3D FRONT Dataset Integration for LP2

이 문서는 3D FRONT 데이터셋을 LP2의 trajectory prediction 메커니즘에 적용하는 방법을 설명합니다.

## 목차
1. [개요](#개요)
2. [데이터 형식 요구사항](#데이터-형식-요구사항)
3. [변환 프로세스](#변환-프로세스)
4. [사용 방법](#사용-방법)
5. [트러블슈팅](#트러블슈팅)

## 개요

LP2는 3D Dynamic Scene Graphs를 사용하여 실내 환경에서 장기 인간 궤적을 예측합니다. 3D FRONT 데이터셋을 사용하려면 다음 세 가지 파일이 필요합니다:

1. **Scene Graph** (Spark-DSG JSON 형식)
2. **Room Labels** (JSON 형식)
3. **Trajectory Data** (CSV 형식)

## 데이터 형식 요구사항

### 1. Scene Graph (Spark-DSG 형식)

Spark-DSG는 계층적 3D scene graph로 다음 layers를 포함합니다:

- **ROOMS Layer (4)**: 방 노드들
  - 속성: `name` (semantic label), `position` (중심점)

- **PLACES Layer (3)**: 공간 위치 노드들 (rooms와 objects 연결)
  - 속성: `name`, `position`

- **OBJECTS Layer (2)**: 가구 객체 노드들
  - 속성: `name` (semantic label), `position`

- **Interlayer Edges**: Rooms ↔ Places ↔ Objects

### 2. Room Labels JSON

```json
{
  "0": "living room",
  "1": "bedroom",
  "2": "kitchen",
  "3": "bathroom"
}
```

Room ID (정수)를 semantic label (문자열)로 매핑합니다.

### 3. Trajectory CSV

각 trajectory는 CSV 파일로 저장되며, 다음 컬럼을 포함합니다:

```csv
,t,x,y,z,room_id,interaction_id,action,object
0,0.0,1.5,2.3,0.0,0,0,standing,sofa
1,1.0,1.6,2.4,0.0,0,0,walking,
2,2.0,1.8,2.6,0.0,0,1,sitting,chair
...
```

필수 컬럼:
- `t`: 시간 (초)
- `x, y, z`: 3D 위치
- `room_id`: 방 ID (room_labels.json의 키와 일치)
- `interaction_id`: 상호작용 ID (연속된 상호작용은 같은 ID)

선택 컬럼:
- `action`: 행동 레이블 (예: walking, sitting, standing)
- `object`: 상호작용하는 객체 (예: chair, table)

## 변환 프로세스

### Step 1: 3D FRONT Scene을 Spark-DSG로 변환

제공된 변환 스크립트를 사용합니다:

```bash
python scripts/convert_3dfront_to_dsg.py \
    --input_json path/to/3dfront_scene.json \
    --output_dir data/3dfront/scene_001/ \
    --scene_id scene_001
```

**출력 파일:**
- `scene_001_scene_graph.json` - Spark-DSG 형식의 scene graph
- `scene_001_room_labels.json` - Room labels

**변환 과정:**

1. **Rooms 파싱**: 3D FRONT JSON의 `room` 배열에서 방 정보 추출
2. **Room 타입 매핑**: 3D FRONT room type → semantic label 변환
   ```
   "LivingRoom" → "living room"
   "MasterRoom" → "bedroom"
   "Kitchen" → "kitchen"
   등등...
   ```
3. **Furniture 파싱**: 각 방의 `children` 필드에서 가구 정보 추출
4. **Places 생성**: 각 가구 근처에 place nodes 자동 생성
5. **Edges 생성**: Room → Place → Object 연결 생성

### Step 2: Trajectory 데이터 준비

#### Option A: 기존 Trajectory 데이터가 있는 경우

Trajectory 데이터를 LP2 CSV 형식으로 변환:

```bash
python scripts/convert_trajectory_to_lp2.py \
    --input_trajectory path/to/your_trajectory.txt \
    --scene_graph data/3dfront/scene_001/scene_001_scene_graph.json \
    --output_dir data/3dfront/scene_001/trajectories/ \
    --trajectory_id traj_001
```

#### Option B: Synthetic Trajectory 생성

Trajectory 데이터가 없는 경우, scene graph 기반으로 synthetic trajectory 생성:

```bash
python scripts/generate_synthetic_trajectories.py \
    --scene_graph data/3dfront/scene_001/scene_001_scene_graph.json \
    --room_labels data/3dfront/scene_001/scene_001_room_labels.json \
    --output_dir data/3dfront/scene_001/trajectories/ \
    --num_trajectories 10 \
    --duration 180.0
```

### Step 3: Scene Config 파일 생성

`config/scene_configs/scene_config_3dfront.yaml` 생성:

```yaml
data:
  scene_graph_path: data/3dfront/scene_001/scene_001_scene_graph.json
  room_labels_path: data/3dfront/scene_001/scene_001_room_labels.json
  trajectory_dir: data/3dfront/scene_001/trajectories

global:
  scene: 3dfront_scene_001
  environment_bounds: [[-10.0, -10.0], [10.0, 10.0]]  # [x_min, y_min], [x_max, y_max]
  bandwidth: 0.8
```

**environment_bounds 설정 방법:**
- Scene의 모든 furniture/room 위치를 포함하는 bounding box
- 자동 계산:
  ```python
  python scripts/compute_scene_bounds.py \
      --scene_graph data/3dfront/scene_001/scene_001_scene_graph.json
  ```

## 사용 방법

### 1. LP2 실행

```bash
python src/lhmp/main.py --run_pipeline "y" \
    --global_config_file "config/global_config.yaml" \
    --method_config_file "config/method_configs/project_config_LP2.yaml" \
    --scene_config_file "config/scene_configs/scene_config_3dfront.yaml"
```

### 2. 결과 확인

출력 디렉토리 (`output/`) 구조:

```
output/
├── 3dfront_scene_001/
│   ├── LP2/
│   │   ├── predictions/          # 예측된 trajectories
│   │   ├── metrics/               # 평가 메트릭
│   │   ├── visualizations/        # 시각화
│   │   └── checkpoints/           # 중간 결과
```

### 3. 시각화

```bash
python scripts/visualize_all.py \
    --methods "LP2" \
    --scenes "3dfront_scene_001" \
    --n_past_interactions 2 \
    --time_lim 60.0
```

## 변환 스크립트 상세 설명

### convert_3dfront_to_dsg.py

**주요 기능:**
- 3D FRONT JSON 파싱
- Room type 및 furniture category 매핑
- Spark-DSG 계층 구조 생성
- Interlayer edges 자동 생성

**커스터마이징:**

스크립트 내의 매핑 딕셔너리를 수정하여 커스터마이징 가능:

```python
# Room type mapping
self.room_type_mapping = {
    "LivingRoom": "living room",
    "CustomRoomType": "custom label",
    # 추가...
}

# Furniture category mapping
self.furniture_category_mapping = {
    "sofa": "sofa",
    "custom_furniture": "custom label",
    # 추가...
}
```

### 3D FRONT JSON 구조 확인

3D FRONT JSON이 표준 형식과 다른 경우, 다음 코드로 구조 확인:

```python
import json

with open("path/to/3dfront_scene.json", 'r') as f:
    data = json.load(f)

# 구조 출력
print("Top-level keys:", data.keys())
print("Scene keys:", data.get("scene", {}).keys())
print("First room:", data["scene"]["room"][0])
print("First furniture:", data["scene"]["furniture"][0])
```

표준과 다른 경우 `convert_3dfront_to_dsg.py`의 파싱 부분 수정:

```python
# line ~140: Adjust field names
scene_info = scene_data.get("scene", scene_data)  # or scene_data["your_key"]
rooms = scene_info.get("room", [])                # or scene_info["your_room_key"]
furniture_list = scene_info.get("furniture", [])  # or scene_info["your_furniture_key"]
```

## 트러블슈팅

### 문제 1: "Scene graph path does not exist"

**원인:** Scene graph 파일 경로가 잘못되었습니다.

**해결:**
```bash
# 경로 확인
ls -la data/3dfront/scene_001/scene_001_scene_graph.json

# config 파일의 scene_graph_path 확인
cat config/scene_configs/scene_config_3dfront.yaml
```

### 문제 2: "No trajectories found"

**원인:** Trajectory 디렉토리가 비어있거나 경로가 잘못되었습니다.

**해결:**
```bash
# Trajectory 파일 확인
ls -la data/3dfront/scene_001/trajectories/*.csv

# 최소 1개 이상의 CSV 파일 필요
```

### 문제 3: "KeyError: 'scene'" 또는 JSON 파싱 오류

**원인:** 3D FRONT JSON 구조가 예상과 다릅니다.

**해결:**
1. JSON 구조 확인 (위의 "3D FRONT JSON 구조 확인" 참조)
2. `convert_3dfront_to_dsg.py` 수정:
   ```python
   # 예시: "scene" 대신 "rooms"라는 키를 사용하는 경우
   # Line ~140 수정
   scene_info = scene_data.get("rooms", scene_data)
   ```

### 문제 4: Room labels mismatch

**원인:** Trajectory CSV의 `room_id`가 room_labels.json의 키와 일치하지 않습니다.

**해결:**
```bash
# Room labels 확인
cat data/3dfront/scene_001/scene_001_room_labels.json

# Trajectory의 room_id 확인
head data/3dfront/scene_001/trajectories/traj_001.csv

# room_id가 room_labels의 키(정수)와 일치해야 함
```

### 문제 5: Furniture not recognized

**원인:** 3D FRONT furniture category가 매핑되지 않았습니다.

**해결:**
```python
# convert_3dfront_to_dsg.py에 카테고리 추가
self.furniture_category_mapping = {
    # 기존 매핑...
    "new_furniture_type": "semantic label",
}
```

### 문제 6: "numpy.float64 is not JSON serializable"

**원인:** Spark-DSG가 최신 버전이 아니거나 numpy 호환성 문제입니다.

**해결:**
```bash
# spark_dsg 업데이트
pip install --upgrade spark-dsg

# 또는 numpy downgrade
pip install numpy==1.24.0
```

## 예제 워크플로우

전체 프로세스 예제:

```bash
# 1. 3D FRONT scene 변환
python scripts/convert_3dfront_to_dsg.py \
    --input_json ~/3DFRONT/scenes/bedroom.json \
    --output_dir data/3dfront/bedroom/ \
    --scene_id bedroom_001

# 2. Synthetic trajectories 생성 (trajectory 데이터가 없는 경우)
python scripts/generate_synthetic_trajectories.py \
    --scene_graph data/3dfront/bedroom/bedroom_001_scene_graph.json \
    --room_labels data/3dfront/bedroom/bedroom_001_room_labels.json \
    --output_dir data/3dfront/bedroom/trajectories/ \
    --num_trajectories 5 \
    --duration 180.0

# 3. Environment bounds 계산
python scripts/compute_scene_bounds.py \
    --scene_graph data/3dfront/bedroom/bedroom_001_scene_graph.json

# 4. Scene config 생성 (출력된 bounds 사용)
cat > config/scene_configs/scene_config_3dfront_bedroom.yaml << EOF
data:
  scene_graph_path: data/3dfront/bedroom/bedroom_001_scene_graph.json
  room_labels_path: data/3dfront/bedroom/bedroom_001_room_labels.json
  trajectory_dir: data/3dfront/bedroom/trajectories

global:
  scene: 3dfront_bedroom_001
  environment_bounds: [[-5.0, -4.0], [8.0, 6.0]]  # From compute_scene_bounds.py
  bandwidth: 0.8
EOF

# 5. LP2 실행
python src/lhmp/main.py --run_pipeline "y" \
    --global_config_file "config/global_config.yaml" \
    --method_config_file "config/method_configs/project_config_LP2.yaml" \
    --scene_config_file "config/scene_configs/scene_config_3dfront_bedroom.yaml"

# 6. 결과 시각화
python scripts/visualize_all.py \
    --methods "LP2" \
    --scenes "3dfront_bedroom_001" \
    --n_past_interactions 2 \
    --time_lim 60.0
```

## 추가 리소스

- **Spark-DSG GitHub**: https://github.com/MIT-SPARK/Spark-DSG
- **LP2 Paper**: https://arxiv.org/abs/2405.00552
- **3D FRONT Dataset**: https://tianchi.aliyun.com/dataset/65347

## 문의 및 기여

Issues 및 Pull Requests는 GitHub 리포지토리에 제출해주세요.
