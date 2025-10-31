# 방 단위 Trajectory 생성 전략

## 문제 정의

### 가지고 있는 데이터
```json
{
  "room": {
    "type": "LivingRoom",
    "bbox": {"min": [x, y, z], "max": [x, y, z]},
    "children": ["furniture_id_1", "furniture_id_2", ...]
  },
  "furniture": [
    {
      "uid": "furniture_id_1",
      "category": "sofa",
      "pos": [x, y, z],
      "rot": [qx, qy, qz, qw],
      "scale": [sx, sy, sz],
      "bbox": {...}
    }
  ]
}
```

### 필요한 것
```csv
t,x,y,z,room_id,interaction_id,action,object
0.0,1.5,2.3,0.0,0,0,interacting_sofa,sofa
1.0,1.5,2.3,0.0,0,0,interacting_sofa,sofa
...
12.5,1.6,2.4,0.0,0,1,walking,
...
15.0,2.0,3.5,0.0,0,1,interacting_table,table
```

## 핵심 아이디어

### 1. **Semantic-based Visiting Order** (의미 기반 방문 순서)

현실에서 사람들은 관련된 가구들을 연속적으로 사용합니다:
- Sofa → Coffee table → TV
- Bed → Nightstand → Wardrobe
- Stove → Sink → Refrigerator

```python
semantic_relationships = {
    "sofa": ["coffee_table", "tv_stand", "side_table"],
    "bed": ["nightstand", "dresser", "wardrobe"],
    "stove": ["sink", "refrigerator", "counter"],
    ...
}
```

**알고리즘:**
1. 임의의 가구에서 시작
2. 현재 가구와 의미적으로 연관된 가구 찾기
3. 그 중 가장 가까운 것 선택
4. 반복

**장점:**
- 현실적인 행동 패턴
- LP2 LLM이 예측하기 쉬움 (의미적 관계 학습됨)

### 2. **Spatial-based Path Planning** (공간 기반 경로 계획)

가구를 피해서 걷는 경로 생성:

```python
# 방 안의 accessible positions 계산
grid_resolution = 0.5  # 50cm grid
for each grid point:
    if distance_to_any_furniture < clearance:
        mark as blocked
    else:
        mark as accessible
```

**경로 생성:**
- Cubic spline interpolation으로 부드러운 곡선
- 중간 control points에 랜덤 offset → 자연스러운 궤적
- 가구 주변 0.5m clearance

```python
def generate_smooth_path(start, end):
    # Control points: start -> midpoint(with offset) -> end
    direction = end - start
    midpoint = start + 0.5 * direction

    # Add perpendicular offset
    perpendicular = [-direction[1], direction[0], 0]
    offset = perpendicular * random(-0.3, 0.3)

    control_points = [start, midpoint + offset, end]

    # Cubic spline through control points
    return cubic_spline(control_points, num_waypoints=20)
```

### 3. **4가지 Visiting Strategies**

#### Strategy 1: **Semantic** (추천)
- 의미적으로 연관된 가구 우선
- 현실적이고 자연스러운 순서
- LP2가 예측하기 쉬움

```python
Current: sofa
  ↓ (semantic relationship)
Next candidates: [coffee_table, tv_stand, side_table]
  ↓ (choose closest)
Next: coffee_table
```

#### Strategy 2: **Spatial**
- 단순히 가장 가까운 가구로 이동
- Greedy nearest-neighbor
- 짧은 이동 거리

```python
Current position: [1.5, 2.3]
  ↓ (compute distances)
Nearest furniture: table at [2.0, 2.5]
  ↓
Next: table
```

#### Strategy 3: **Circular**
- 방을 한바퀴 도는 패턴
- 방 중심에서 angle 기준 정렬
- 순찰 패턴

```python
# Compute angles from room center
for furniture in room:
    angle = atan2(y - center_y, x - center_x)

# Sort by angle
sorted_furniture = sort_by_angle(furniture_list)

# Visit in circular order
```

#### Strategy 4: **Random**
- 랜덤 순서
- Baseline 비교용

### 4. **Interaction Duration Modeling**

현실적인 interaction 시간:

```python
# 가구 타입별 평균 interaction duration
duration_by_type = {
    "sofa": (10.0, 60.0),      # 앉아서 쉬기
    "tv": (20.0, 120.0),        # TV 시청
    "table": (5.0, 30.0),       # 물건 올려놓기
    "bed": (30.0, 300.0),       # 침대에 눕기
    "stove": (5.0, 15.0),       # 요리
    "sink": (2.0, 10.0),        # 설거지
    "refrigerator": (5.0, 20.0),# 음식 꺼내기
}

# Use random sampling within range
duration = random.uniform(min_duration, max_duration)
```

### 5. **Past/Future Trajectory Split**

LP2를 위한 train/test split:

```python
# LP2 requirements:
# - Past: 2 interactions + 2 seconds after
# - Future: 60+ seconds for prediction

def split_trajectory(full_trajectory, split_at_interaction=2):
    # Find end of interaction #2
    interaction_end = full_trajectory[
        full_trajectory["interaction_id"] == split_at_interaction
    ].iloc[-1]["t"]

    # Add 2 seconds
    split_time = interaction_end + 2.0

    past = full_trajectory[full_trajectory["t"] <= split_time]
    future = full_trajectory[full_trajectory["t"] > split_time]

    return past, future
```

## 구현 예제

### Example 1: Living Room Trajectory

```python
room_data = {
    "type": "LivingRoom",
    "bbox": {"min": [-5, -4, 0], "max": [5, 4, 3]},
    "children": ["furn_1", "furn_2", "furn_3", "furn_4"]
}

furniture_data = [
    {"uid": "furn_1", "category": "sofa", "pos": [0, 0, 0]},
    {"uid": "furn_2", "category": "coffee_table", "pos": [0, 1.5, 0]},
    {"uid": "furn_3", "category": "tv_stand", "pos": [0, 3, 0]},
    {"uid": "furn_4", "category": "armchair", "pos": [2, 0, 0]},
]

generator = RoomTrajectoryGenerator(room_data, furniture_data)

# Generate trajectory with semantic strategy
past_traj, full_traj = generator.generate_trajectory(
    num_interactions=4,
    total_duration=120.0,
    strategy="semantic",
    split_at_interaction=2
)

# Result:
# Interaction sequence: sofa -> coffee_table -> tv_stand -> armchair
# Past trajectory: sofa, coffee_table (+ 2s walking)
# Future trajectory: tv_stand, armchair
```

### Example 2: Multiple Trajectories with Different Strategies

```bash
# Semantic strategy (realistic)
python scripts/generate_room_trajectory.py \
    --room_json bedroom.json \
    --output_dir data/bedroom/semantic/ \
    --num_trajectories 20 \
    --strategy semantic \
    --num_interactions 5

# Circular strategy (patrol pattern)
python scripts/generate_room_trajectory.py \
    --room_json bedroom.json \
    --output_dir data/bedroom/circular/ \
    --num_trajectories 20 \
    --strategy circular \
    --num_interactions 5

# Spatial strategy (nearest neighbor)
python scripts/generate_room_trajectory.py \
    --room_json bedroom.json \
    --output_dir data/bedroom/spatial/ \
    --num_trajectories 20 \
    --strategy spatial \
    --num_interactions 5
```

## 고급 개선 아이디어

### 1. **Activity-based Trajectories**

특정 activity를 수행하는 trajectory 생성:

```python
activities = {
    "morning_routine": ["bed", "wardrobe", "bathroom", "kitchen"],
    "relaxing": ["sofa", "tv", "coffee_table"],
    "cooking": ["refrigerator", "counter", "stove", "sink"],
    "working": ["desk", "chair", "bookshelf"],
}

# Generate trajectory following activity pattern
def generate_activity_trajectory(activity_name):
    furniture_sequence = activities[activity_name]
    # ... generate path
```

### 2. **Time-of-day Variations**

시간대별 다른 행동 패턴:

```python
if time_of_day == "morning":
    # More interactions with: bed, bathroom, kitchen
    weights = {"bed": 0.3, "bathroom": 0.3, "kitchen": 0.3}
elif time_of_day == "evening":
    # More interactions with: sofa, tv, dining
    weights = {"sofa": 0.4, "tv": 0.3, "dining_table": 0.2}
```

### 3. **Person-specific Behaviors**

다른 사람들의 행동 패턴:

```python
person_profiles = {
    "adult_worker": {
        "preferred_furniture": ["desk", "chair", "coffee_machine"],
        "avg_interaction_duration": (10, 30),
    },
    "elderly": {
        "walking_speed": 0.7,  # slower
        "preferred_furniture": ["sofa", "armchair", "tv"],
        "avg_interaction_duration": (20, 60),  # longer
    },
    "child": {
        "walking_speed": 1.2,  # faster
        "preferred_furniture": ["toys", "bed", "desk"],
        "avg_interaction_duration": (5, 15),  # shorter
    }
}
```

### 4. **Multi-person Trajectories**

여러 사람이 동시에 움직이는 trajectory:

```python
def generate_multi_person_trajectory(room_data, num_people=2):
    # Person 1: semantic strategy
    # Person 2: spatial strategy
    # Avoid collision between persons
    # Consider social interactions (e.g., dining together)
```

### 5. **Obstacle-aware Path Planning**

더 정교한 경로 생성:

```python
# Use A* or RRT algorithm
def plan_path_with_astar(start, goal, obstacles):
    # Build occupancy grid
    grid = create_occupancy_grid(room_bbox, obstacles)

    # Run A* search
    path = astar_search(grid, start, goal)

    # Smooth path with splines
    smooth_path = smooth_with_spline(path)

    return smooth_path
```

## 사용 시나리오

### Scenario 1: LP2 Input 생성

```bash
# 1. Generate room trajectories
python scripts/generate_room_trajectory.py \
    --room_json my_room.json \
    --output_dir data/my_room/trajectories/ \
    --num_trajectories 50 \
    --strategy semantic

# 2. Convert room to scene graph
python scripts/convert_3dfront_to_dsg.py \
    --input_json my_room.json \
    --output_dir data/my_room/ \
    --scene_id my_room

# 3. Create scene config
# Use data/my_room/trajectories/past/ as trajectory_dir

# 4. Run LP2
python src/lhmp/main.py --run_pipeline "y" \
    --scene_config_file config/scene_configs/my_room.yaml
```

### Scenario 2: 여러 전략 비교

```bash
# Generate with all strategies
for strategy in semantic spatial circular random; do
    python scripts/generate_room_trajectory.py \
        --room_json bedroom.json \
        --output_dir data/bedroom_${strategy}/ \
        --num_trajectories 20 \
        --strategy $strategy
done

# Run LP2 on each
for strategy in semantic spatial circular random; do
    python src/lhmp/main.py --run_pipeline "y" \
        --scene_config_file config/scene_configs/bedroom_${strategy}.yaml
done

# Compare prediction accuracy
python scripts/compare_strategies.py
```

### Scenario 3: 데이터 증강

```bash
# Generate large dataset with variations
python scripts/generate_room_trajectory.py \
    --room_json bedroom.json \
    --output_dir data/bedroom_augmented/ \
    --num_trajectories 100 \
    --strategy semantic \
    --num_interactions 6

# Vary interaction durations
for duration in 60 90 120 180; do
    python scripts/generate_room_trajectory.py \
        --room_json bedroom.json \
        --output_dir data/bedroom_dur${duration}/ \
        --num_trajectories 20 \
        --duration $duration
done
```

## 평가 메트릭

생성된 trajectory의 품질 평가:

```python
# 1. Spatial coverage
def compute_coverage(trajectories, room_bbox):
    visited_positions = extract_positions(trajectories)
    coverage_ratio = len(unique_grid_cells(visited_positions)) / total_grid_cells
    return coverage_ratio

# 2. Path efficiency
def compute_path_efficiency(trajectory):
    total_distance = compute_path_length(trajectory)
    direct_distance = distance(start, end)
    efficiency = direct_distance / total_distance
    return efficiency

# 3. Interaction realism
def compute_interaction_realism(trajectory):
    for interaction in extract_interactions(trajectory):
        # Check duration is reasonable
        # Check furniture type matches action
        # Check spatial continuity
    return realism_score

# 4. Diversity
def compute_diversity(trajectories):
    # Measure variety in:
    # - Visiting orders
    # - Path shapes
    # - Interaction durations
    return diversity_score
```

## 트러블슈팅

### 문제 1: 가구가 너무 적음

```python
# Solution: Add virtual waypoints
def add_virtual_waypoints(furniture_list, num_waypoints=5):
    # Add empty space positions as interaction points
    for i in range(num_waypoints):
        virtual_pos = sample_random_accessible_position()
        furniture_list.append({
            "category": "waypoint",
            "pos": virtual_pos
        })
```

### 문제 2: 경로가 가구와 충돌

```python
# Solution: Increase clearance
generator = RoomTrajectoryGenerator(
    room_data,
    furniture_data,
    furniture_clearance=0.8  # Increase from 0.5 to 0.8
)
```

### 문제 3: Interaction duration이 너무 짧음

```python
# Solution: Adjust duration range
generator = RoomTrajectoryGenerator(
    room_data,
    furniture_data,
    interaction_duration_range=(10.0, 30.0)  # Longer interactions
)
```

### 문제 4: Trajectory가 부자연스러움

```python
# Solution: Use semantic strategy + more control points
generator.generate_trajectory(
    strategy="semantic",  # More realistic ordering
    num_interactions=5,   # Not too many jumps
)

# And increase path smoothness
def generate_smooth_path(start, end, num_waypoints=50):  # More waypoints
    # More control points
    num_control = 3  # Instead of 2
    ...
```

## 요약

방 단위 trajectory 생성의 핵심:

1. **Semantic relationships**: 의미적으로 연관된 가구 방문
2. **Spatial planning**: 가구를 피해 부드러운 경로 생성
3. **Realistic durations**: 가구 타입별 적절한 interaction 시간
4. **Multiple strategies**: 다양한 방문 순서 전략 제공
5. **Train/test split**: LP2 요구사항에 맞는 과거/미래 분리

이 방법으로 3D FRONT 방 데이터에서 realistic하고 LP2가 잘 예측할 수 있는 trajectory를 생성할 수 있습니다!
