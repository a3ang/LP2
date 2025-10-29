#!/bin/bash
# Example workflow for converting 3D FRONT scenes to LP2 format
# and running trajectory prediction
#
# This script demonstrates the complete pipeline:
# 1. Convert 3D FRONT scene to Spark-DSG format
# 2. Generate synthetic trajectories
# 3. Compute scene bounds
# 4. Create scene config
# 5. Run LP2 prediction
#
# Usage:
#   bash scripts/example_3dfront_workflow.sh path/to/3dfront_scene.json bedroom_001

set -e  # Exit on error

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <3dfront_json_path> <scene_id>"
    echo "Example: $0 ~/3DFRONT/bedroom.json bedroom_001"
    exit 1
fi

INPUT_JSON=$1
SCENE_ID=$2
OUTPUT_BASE="data/3dfront/${SCENE_ID}"

echo "=========================================="
echo "3D FRONT to LP2 Conversion Workflow"
echo "=========================================="
echo "Input JSON: ${INPUT_JSON}"
echo "Scene ID: ${SCENE_ID}"
echo "Output directory: ${OUTPUT_BASE}"
echo ""

# Step 1: Convert 3D FRONT scene to Spark-DSG
echo "Step 1: Converting 3D FRONT scene to Spark-DSG format..."
python scripts/convert_3dfront_to_dsg.py \
    --input_json "${INPUT_JSON}" \
    --output_dir "${OUTPUT_BASE}" \
    --scene_id "${SCENE_ID}"

if [ $? -ne 0 ]; then
    echo "Error: Scene conversion failed"
    exit 1
fi
echo "✓ Scene conversion complete"
echo ""

# Step 2: Generate synthetic trajectories
echo "Step 2: Generating synthetic trajectories..."
python scripts/generate_synthetic_trajectories.py \
    --scene_graph "${OUTPUT_BASE}/${SCENE_ID}_scene_graph.json" \
    --room_labels "${OUTPUT_BASE}/${SCENE_ID}_room_labels.json" \
    --output_dir "${OUTPUT_BASE}/trajectories" \
    --num_trajectories 10 \
    --duration 180.0 \
    --min_interactions 3 \
    --max_interactions 7 \
    --walking_speed 1.0 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "Error: Trajectory generation failed"
    exit 1
fi
echo "✓ Trajectory generation complete"
echo ""

# Step 3: Compute scene bounds
echo "Step 3: Computing scene bounds..."
BOUNDS_OUTPUT=$(python scripts/compute_scene_bounds.py \
    --scene_graph "${OUTPUT_BASE}/${SCENE_ID}_scene_graph.json" \
    --padding 1.0)

echo "${BOUNDS_OUTPUT}"

# Extract bounds from output (last line with format: environment_bounds: [...])
BOUNDS=$(echo "${BOUNDS_OUTPUT}" | grep "environment_bounds:" | sed 's/environment_bounds: //')

if [ -z "${BOUNDS}" ]; then
    echo "Warning: Could not extract bounds automatically, using default"
    BOUNDS="[[-10.0, -10.0], [10.0, 10.0]]"
fi

echo "✓ Scene bounds computed: ${BOUNDS}"
echo ""

# Step 4: Create scene config file
CONFIG_FILE="config/scene_configs/scene_config_3dfront_${SCENE_ID}.yaml"
echo "Step 4: Creating scene config file: ${CONFIG_FILE}"

cat > "${CONFIG_FILE}" << EOF
# Auto-generated config for 3D FRONT scene: ${SCENE_ID}
# Generated on: $(date)

data:
  scene_graph_path: ${OUTPUT_BASE}/${SCENE_ID}_scene_graph.json
  room_labels_path: ${OUTPUT_BASE}/${SCENE_ID}_room_labels.json
  trajectory_dir: ${OUTPUT_BASE}/trajectories

global:
  scene: 3dfront_${SCENE_ID}
  environment_bounds: ${BOUNDS}
  bandwidth: 0.8
EOF

echo "✓ Config file created"
echo ""

# Step 5: Display summary
echo "=========================================="
echo "Conversion Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  Scene graph: ${OUTPUT_BASE}/${SCENE_ID}_scene_graph.json"
echo "  Room labels: ${OUTPUT_BASE}/${SCENE_ID}_room_labels.json"
echo "  Trajectories: ${OUTPUT_BASE}/trajectories/"
echo "  Config file: ${CONFIG_FILE}"
echo ""
echo "To run LP2 prediction:"
echo "  python src/lhmp/main.py --run_pipeline \"y\" \\"
echo "      --global_config_file \"config/global_config.yaml\" \\"
echo "      --method_config_file \"config/method_configs/project_config_LP2.yaml\" \\"
echo "      --scene_config_file \"${CONFIG_FILE}\""
echo ""
echo "To visualize results:"
echo "  python scripts/visualize_all.py \\"
echo "      --methods \"LP2\" \\"
echo "      --scenes \"3dfront_${SCENE_ID}\" \\"
echo "      --n_past_interactions 2 \\"
echo "      --time_lim 60.0"
echo ""
