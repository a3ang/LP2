#!/usr/bin/env python3
"""
Generate synthetic trajectories for LP2 from scene graphs

This script generates synthetic human trajectories by:
1. Loading a scene graph
2. Randomly selecting sequences of objects to interact with
3. Generating smooth paths between objects
4. Creating CSV files in LP2 format

Usage:
    python scripts/generate_synthetic_trajectories.py \
        --scene_graph data/3dfront/scene_001/scene_001_scene_graph.json \
        --room_labels data/3dfront/scene_001/scene_001_room_labels.json \
        --output_dir data/3dfront/scene_001/trajectories/ \
        --num_trajectories 10 \
        --duration 180.0

Author: Claude Code
Date: 2025-10-29
"""

import argparse
import json
import numpy as np
import pandas as pd
import spark_dsg as dsg
from pathlib import Path
from typing import Dict, List, Tuple
import random
from scipy.interpolate import interp1d


class SyntheticTrajectoryGenerator:
    """Generate synthetic trajectories from scene graphs"""

    def __init__(
        self,
        scene_graph: dsg.DynamicSceneGraph,
        room_labels: Dict[int, str],
        walking_speed: float = 1.0,
        interaction_duration_range: Tuple[float, float] = (5.0, 15.0),
    ):
        """
        Initialize the generator

        Args:
            scene_graph: Spark-DSG scene graph
            room_labels: Dictionary mapping room IDs to labels
            walking_speed: Average walking speed in m/s
            interaction_duration_range: Min/max interaction duration in seconds
        """
        self.scene_graph = scene_graph
        self.room_labels = room_labels
        self.walking_speed = walking_speed
        self.interaction_duration_range = interaction_duration_range

        # Layer IDs
        self.OBJECTS_LAYER = 2
        self.PLACES_LAYER = 3
        self.ROOMS_LAYER = 4

        # Extract objects and their properties
        self.objects = self._extract_objects()
        self.places = self._extract_places()

    def _extract_objects(self) -> List[Dict]:
        """Extract object information from scene graph"""
        objects = []
        objects_layer = self.scene_graph.get_layer(self.OBJECTS_LAYER)

        for node_id in objects_layer.nodes:
            node = self.scene_graph.get_node(node_id)
            position = node.attributes.position
            semantic_label = node.attributes.name

            # Find room ID through interlayer edges
            room_id = self._find_room_for_node(node_id)

            objects.append({
                "id": node_id,
                "semantic_label": semantic_label,
                "position": position,
                "room_id": room_id,
            })

        return objects

    def _extract_places(self) -> List[Dict]:
        """Extract place information from scene graph"""
        places = []
        places_layer = self.scene_graph.get_layer(self.PLACES_LAYER)

        for node_id in places_layer.nodes:
            node = self.scene_graph.get_node(node_id)
            position = node.attributes.position
            room_id = self._find_room_for_node(node_id)

            places.append({
                "id": node_id,
                "position": position,
                "room_id": room_id,
            })

        return places

    def _find_room_for_node(self, node_id: int) -> int:
        """Find the room ID for a given node through interlayer edges"""
        # Traverse interlayer edges to find the room
        for edge in self.scene_graph.interlayer_edges:
            if edge.target == node_id:
                # Check if source is a room
                source_node = self.scene_graph.get_node(edge.source)
                if source_node.layer.id == self.ROOMS_LAYER:
                    # Extract room category ID from node symbol
                    from spark_dsg._dsg_bindings import NodeSymbol
                    symbol = NodeSymbol(edge.source)
                    return symbol.category_id
                # Otherwise, recursively check the source
                return self._find_room_for_node(edge.source)
        return -1  # Unknown room

    def generate_trajectory(
        self,
        num_interactions: int = 5,
        duration: float = 180.0,
    ) -> pd.DataFrame:
        """
        Generate a single synthetic trajectory

        Args:
            num_interactions: Number of object interactions
            duration: Total trajectory duration in seconds

        Returns:
            DataFrame with trajectory in LP2 CSV format
        """
        if len(self.objects) < num_interactions:
            num_interactions = len(self.objects)

        # Sample random objects for interaction
        interaction_objects = random.sample(self.objects, num_interactions)

        # Generate trajectory segments
        trajectory_data = []
        current_time = 0.0
        interaction_id = 0

        for obj_idx, obj in enumerate(interaction_objects):
            # Interaction phase
            interaction_duration = random.uniform(*self.interaction_duration_range)

            # Add interaction waypoints
            num_interaction_steps = int(interaction_duration * 10)  # 0.1s timestep
            for step in range(num_interaction_steps):
                t = current_time + step * 0.1
                if t > duration:
                    break

                # Add small random motion during interaction (standing/fidgeting)
                noise = np.random.normal(0, 0.05, 3)  # 5cm std dev
                position = obj["position"] + noise

                trajectory_data.append({
                    "t": t,
                    "x": position[0],
                    "y": position[1],
                    "z": position[2],
                    "room_id": obj["room_id"],
                    "interaction_id": interaction_id,
                    "action": f"interacting_{obj['semantic_label']}",
                    "object": obj["semantic_label"],
                })

            current_time += interaction_duration
            interaction_id += 1

            if current_time > duration or obj_idx == len(interaction_objects) - 1:
                break

            # Walking phase to next object
            next_obj = interaction_objects[obj_idx + 1]
            distance = np.linalg.norm(obj["position"] - next_obj["position"])
            walking_duration = distance / self.walking_speed

            # Generate smooth path
            path_waypoints = self._generate_smooth_path(
                obj["position"], next_obj["position"], num_waypoints=20
            )

            num_walking_steps = int(walking_duration * 10)
            for step in range(num_walking_steps):
                t = current_time + step * 0.1
                if t > duration:
                    break

                # Interpolate position along path
                progress = step / max(num_walking_steps - 1, 1)
                waypoint_idx = int(progress * (len(path_waypoints) - 1))
                position = path_waypoints[waypoint_idx]

                # Determine current room (simplified: check which object is closer)
                dist_to_current = np.linalg.norm(position - obj["position"])
                dist_to_next = np.linalg.norm(position - next_obj["position"])
                room_id = obj["room_id"] if dist_to_current < dist_to_next else next_obj["room_id"]

                trajectory_data.append({
                    "t": t,
                    "x": position[0],
                    "y": position[1],
                    "z": position[2],
                    "room_id": room_id,
                    "interaction_id": interaction_id,
                    "action": "walking",
                    "object": "",
                })

            current_time += walking_duration

        # Convert to DataFrame
        df = pd.DataFrame(trajectory_data)
        return df

    def _generate_smooth_path(
        self, start: np.ndarray, end: np.ndarray, num_waypoints: int = 20
    ) -> np.ndarray:
        """
        Generate a smooth path between two points using cubic interpolation

        Args:
            start: Start position [x, y, z]
            end: End position [x, y, z]
            num_waypoints: Number of waypoints along path

        Returns:
            Array of shape (num_waypoints, 3) with smooth path
        """
        # Generate control points (start, midpoint with offset, end)
        midpoint = (start + end) / 2
        # Add random offset perpendicular to direct path (more natural)
        direction = end - start
        perpendicular = np.array([-direction[1], direction[0], 0])
        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-8)
        offset = perpendicular * random.uniform(-0.5, 0.5)
        midpoint += offset

        # Cubic interpolation through control points
        control_points = np.array([start, midpoint, end])
        t_control = np.linspace(0, 1, len(control_points))
        t_waypoints = np.linspace(0, 1, num_waypoints)

        # Interpolate each dimension
        path = np.zeros((num_waypoints, 3))
        for dim in range(3):
            interp_func = interp1d(
                t_control, control_points[:, dim], kind='quadratic'
            )
            path[:, dim] = interp_func(t_waypoints)

        return path

    def generate_multiple_trajectories(
        self,
        num_trajectories: int,
        num_interactions_range: Tuple[int, int] = (3, 7),
        duration: float = 180.0,
    ) -> List[pd.DataFrame]:
        """
        Generate multiple synthetic trajectories

        Args:
            num_trajectories: Number of trajectories to generate
            num_interactions_range: Min/max number of interactions per trajectory
            duration: Trajectory duration in seconds

        Returns:
            List of trajectory DataFrames
        """
        trajectories = []
        for i in range(num_trajectories):
            num_interactions = random.randint(*num_interactions_range)
            traj = self.generate_trajectory(
                num_interactions=num_interactions, duration=duration
            )
            trajectories.append(traj)
            print(f"Generated trajectory {i+1}/{num_trajectories} "
                  f"({len(traj)} timesteps, {num_interactions} interactions)")

        return trajectories

    def save_trajectories(
        self, trajectories: List[pd.DataFrame], output_dir: str, prefix: str = "traj"
    ):
        """
        Save trajectories to CSV files

        Args:
            trajectories: List of trajectory DataFrames
            output_dir: Output directory
            prefix: Filename prefix (default: "traj")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, traj in enumerate(trajectories):
            filename = output_dir / f"{prefix}_{i:04d}.csv"
            traj.to_csv(filename)
            print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic trajectories for LP2"
    )
    parser.add_argument(
        "--scene_graph",
        type=str,
        required=True,
        help="Path to scene graph JSON file"
    )
    parser.add_argument(
        "--room_labels",
        type=str,
        required=True,
        help="Path to room labels JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for trajectory CSV files"
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=10,
        help="Number of trajectories to generate (default: 10)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=180.0,
        help="Trajectory duration in seconds (default: 180.0)"
    )
    parser.add_argument(
        "--min_interactions",
        type=int,
        default=3,
        help="Minimum number of interactions per trajectory (default: 3)"
    )
    parser.add_argument(
        "--max_interactions",
        type=int,
        default=7,
        help="Maximum number of interactions per trajectory (default: 7)"
    )
    parser.add_argument(
        "--walking_speed",
        type=float,
        default=1.0,
        help="Walking speed in m/s (default: 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load scene graph
    print(f"Loading scene graph: {args.scene_graph}")
    scene_graph = dsg.DynamicSceneGraph.load(args.scene_graph)

    # Load room labels
    print(f"Loading room labels: {args.room_labels}")
    with open(args.room_labels, 'r') as f:
        room_labels = json.load(f)
        # Convert string keys to int
        room_labels = {int(k): v for k, v in room_labels.items()}

    # Create generator
    generator = SyntheticTrajectoryGenerator(
        scene_graph=scene_graph,
        room_labels=room_labels,
        walking_speed=args.walking_speed,
    )

    print(f"\nGenerating {args.num_trajectories} trajectories...")
    print(f"  Duration: {args.duration}s")
    print(f"  Interactions per trajectory: {args.min_interactions}-{args.max_interactions}")
    print(f"  Walking speed: {args.walking_speed} m/s")

    # Generate trajectories
    trajectories = generator.generate_multiple_trajectories(
        num_trajectories=args.num_trajectories,
        num_interactions_range=(args.min_interactions, args.max_interactions),
        duration=args.duration,
    )

    # Save trajectories
    print(f"\nSaving trajectories to: {args.output_dir}")
    generator.save_trajectories(trajectories, args.output_dir)

    print(f"\nGeneration complete! Created {len(trajectories)} trajectories.")


if __name__ == "__main__":
    main()
