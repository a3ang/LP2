#!/usr/bin/env python3
"""
Compute environment bounds for a scene graph

This script analyzes a scene graph and computes the bounding box
that encompasses all objects and places. The output can be used
for the 'environment_bounds' parameter in scene config files.

Usage:
    python scripts/compute_scene_bounds.py \
        --scene_graph data/3dfront/scene_001/scene_001_scene_graph.json \
        --padding 1.0

Author: Claude Code
Date: 2025-10-29
"""

import argparse
import numpy as np
import spark_dsg as dsg
from pathlib import Path


def compute_scene_bounds(scene_graph_path: str, padding: float = 1.0):
    """
    Compute environment bounds for a scene graph

    Args:
        scene_graph_path: Path to scene graph JSON file
        padding: Additional padding around bounds in meters

    Returns:
        Tuple of ((x_min, y_min), (x_max, y_max))
    """
    # Load scene graph
    print(f"Loading scene graph: {scene_graph_path}")
    scene_graph = dsg.DynamicSceneGraph.load(scene_graph_path)

    # Layer IDs
    OBJECTS_LAYER = 2
    PLACES_LAYER = 3
    ROOMS_LAYER = 4

    # Collect all positions
    all_positions = []

    # Get objects layer
    try:
        objects_layer = scene_graph.get_layer(OBJECTS_LAYER)
        for node_id in objects_layer.nodes:
            node = scene_graph.get_node(node_id)
            pos = node.attributes.position
            all_positions.append(pos[:2])  # Only x, y
        print(f"  Objects: {len(objects_layer.nodes)} nodes")
    except Exception as e:
        print(f"  Warning: Could not load objects layer: {e}")

    # Get places layer
    try:
        places_layer = scene_graph.get_layer(PLACES_LAYER)
        for node_id in places_layer.nodes:
            node = scene_graph.get_node(node_id)
            pos = node.attributes.position
            all_positions.append(pos[:2])  # Only x, y
        print(f"  Places: {len(places_layer.nodes)} nodes")
    except Exception as e:
        print(f"  Warning: Could not load places layer: {e}")

    # Get rooms layer
    try:
        rooms_layer = scene_graph.get_layer(ROOMS_LAYER)
        for node_id in rooms_layer.nodes:
            node = scene_graph.get_node(node_id)
            pos = node.attributes.position
            all_positions.append(pos[:2])  # Only x, y
        print(f"  Rooms: {len(rooms_layer.nodes)} nodes")
    except Exception as e:
        print(f"  Warning: Could not load rooms layer: {e}")

    if not all_positions:
        print("Error: No positions found in scene graph!")
        return None

    # Convert to numpy array
    all_positions = np.array(all_positions)

    # Compute bounds
    min_pos = all_positions.min(axis=0)
    max_pos = all_positions.max(axis=0)

    # Add padding
    min_pos -= padding
    max_pos += padding

    # Round to 1 decimal place
    min_pos = np.round(min_pos, 1)
    max_pos = np.round(max_pos, 1)

    return (min_pos.tolist(), max_pos.tolist())


def main():
    parser = argparse.ArgumentParser(
        description="Compute environment bounds for a scene graph"
    )
    parser.add_argument(
        "--scene_graph",
        type=str,
        required=True,
        help="Path to scene graph JSON file"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.0,
        help="Padding around bounds in meters (default: 1.0)"
    )

    args = parser.parse_args()

    # Compute bounds
    bounds = compute_scene_bounds(args.scene_graph, args.padding)

    if bounds is not None:
        min_pos, max_pos = bounds
        print("\n" + "="*60)
        print("Environment Bounds (with {:.1f}m padding):".format(args.padding))
        print("="*60)
        print(f"  X range: [{min_pos[0]:.1f}, {max_pos[0]:.1f}]  (width: {max_pos[0] - min_pos[0]:.1f}m)")
        print(f"  Y range: [{min_pos[1]:.1f}, {max_pos[1]:.1f}]  (depth: {max_pos[1] - min_pos[1]:.1f}m)")
        print("\nYAML config format:")
        print("-"*60)
        print(f"environment_bounds: [{min_pos}, {max_pos}]")
        print("-"*60)
    else:
        print("Failed to compute bounds")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
