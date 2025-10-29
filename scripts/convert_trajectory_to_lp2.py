#!/usr/bin/env python3
"""
Convert trajectory data to LP2 CSV format

This script converts trajectory data from various formats to the LP2 CSV format.
Supports common trajectory formats and can infer room IDs and interactions
from the scene graph.

Input formats supported:
- Simple XYZ: text file with "t x y z" per line
- JSON: {"trajectory": [[t, x, y, z], ...]}
- CSV: with columns t,x,y,z

Output format (LP2):
- CSV with columns: t, x, y, z, room_id, interaction_id, action, object

Usage:
    python scripts/convert_trajectory_to_lp2.py \
        --input_trajectory path/to/trajectory.txt \
        --input_format xyz \
        --scene_graph data/3dfront/scene_001/scene_001_scene_graph.json \
        --output_dir data/3dfront/scene_001/trajectories/ \
        --trajectory_id traj_001

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
from scipy.spatial import cKDTree


class TrajectoryConverter:
    """Convert trajectory data to LP2 format"""

    def __init__(self, scene_graph: dsg.DynamicSceneGraph):
        """
        Initialize converter

        Args:
            scene_graph: Spark-DSG scene graph for spatial context
        """
        self.scene_graph = scene_graph

        # Layer IDs
        self.OBJECTS_LAYER = 2
        self.PLACES_LAYER = 3
        self.ROOMS_LAYER = 4

        # Build spatial indices
        self._build_spatial_indices()

    def _build_spatial_indices(self):
        """Build KD-trees for efficient spatial queries"""
        # Extract objects
        self.objects = []
        objects_layer = self.scene_graph.get_layer(self.OBJECTS_LAYER)
        for node_id in objects_layer.nodes:
            node = self.scene_graph.get_node(node_id)
            self.objects.append({
                "id": node_id,
                "position": node.attributes.position,
                "semantic_label": node.attributes.name,
            })

        # Extract places with room IDs
        self.places = []
        places_layer = self.scene_graph.get_layer(self.PLACES_LAYER)
        for node_id in places_layer.nodes:
            node = self.scene_graph.get_node(node_id)
            room_id = self._find_room_for_place(node_id)
            self.places.append({
                "id": node_id,
                "position": node.attributes.position,
                "room_id": room_id,
            })

        # Build KD-trees
        if self.objects:
            object_positions = np.array([obj["position"] for obj in self.objects])
            self.object_kdtree = cKDTree(object_positions)
        else:
            self.object_kdtree = None

        if self.places:
            place_positions = np.array([place["position"] for place in self.places])
            self.place_kdtree = cKDTree(place_positions)
        else:
            self.place_kdtree = None

        print(f"Built spatial indices: {len(self.objects)} objects, {len(self.places)} places")

    def _find_room_for_place(self, place_id: int) -> int:
        """Find room ID for a place node"""
        for edge in self.scene_graph.interlayer_edges:
            if edge.target == place_id:
                source_node = self.scene_graph.get_node(edge.source)
                if source_node.layer.id == self.ROOMS_LAYER:
                    from spark_dsg._dsg_bindings import NodeSymbol
                    symbol = NodeSymbol(edge.source)
                    return symbol.category_id
        return -1

    def load_trajectory(self, input_path: str, input_format: str) -> pd.DataFrame:
        """
        Load trajectory from file

        Args:
            input_path: Path to trajectory file
            input_format: Format type ('xyz', 'json', 'csv')

        Returns:
            DataFrame with columns [t, x, y, z]
        """
        if input_format == "xyz":
            # Simple text format: t x y z
            data = np.loadtxt(input_path)
            df = pd.DataFrame(data, columns=["t", "x", "y", "z"])

        elif input_format == "json":
            with open(input_path, 'r') as f:
                data = json.load(f)
            # Expect {"trajectory": [[t, x, y, z], ...]}
            traj_data = data.get("trajectory", data.get("data", []))
            df = pd.DataFrame(traj_data, columns=["t", "x", "y", "z"])

        elif input_format == "csv":
            df = pd.read_csv(input_path)
            # Check for required columns
            required = ["t", "x", "y", "z"]
            if not all(col in df.columns for col in required):
                raise ValueError(f"CSV must contain columns: {required}")
            df = df[required]

        else:
            raise ValueError(f"Unsupported format: {input_format}")

        print(f"Loaded trajectory: {len(df)} timesteps")
        return df

    def infer_room_ids(self, trajectory: pd.DataFrame) -> np.ndarray:
        """
        Infer room IDs for trajectory positions

        Args:
            trajectory: DataFrame with x, y, z columns

        Returns:
            Array of room IDs
        """
        if self.place_kdtree is None:
            print("Warning: No places found, using default room ID")
            return np.zeros(len(trajectory), dtype=int)

        positions = trajectory[["x", "y", "z"]].values
        room_ids = []

        for pos in positions:
            # Find nearest place
            _, nearest_idx = self.place_kdtree.query(pos)
            room_id = self.places[nearest_idx]["room_id"]
            room_ids.append(room_id)

        return np.array(room_ids)

    def infer_interactions(
        self,
        trajectory: pd.DataFrame,
        interaction_threshold: float = 0.5,
        min_duration: float = 3.0
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Infer interactions from trajectory

        Detects when the person is near an object and stationary

        Args:
            trajectory: DataFrame with t, x, y, z columns
            interaction_threshold: Distance threshold for interaction (meters)
            min_duration: Minimum interaction duration (seconds)

        Returns:
            Tuple of (interaction_ids array, object_names list)
        """
        if self.object_kdtree is None:
            print("Warning: No objects found, no interactions detected")
            return np.zeros(len(trajectory), dtype=int), [""] * len(trajectory)

        positions = trajectory[["x", "y", "z"]].values
        times = trajectory["t"].values

        # Compute velocity
        velocities = np.zeros(len(trajectory))
        for i in range(1, len(trajectory)):
            dt = times[i] - times[i-1]
            if dt > 0:
                velocities[i] = np.linalg.norm(positions[i] - positions[i-1]) / dt

        # Find nearby objects and low velocity
        interaction_ids = np.zeros(len(trajectory), dtype=int)
        object_names = [""] * len(trajectory)
        current_interaction_id = 0

        in_interaction = False
        interaction_start = 0
        current_object_idx = None

        for i in range(len(trajectory)):
            # Find nearest object
            dist, nearest_obj_idx = self.object_kdtree.query(positions[i])

            # Check if near object and moving slowly
            is_near = dist < interaction_threshold
            is_slow = velocities[i] < 0.3  # 0.3 m/s threshold

            if is_near and is_slow:
                if not in_interaction:
                    # Start new interaction
                    in_interaction = True
                    interaction_start = i
                    current_object_idx = nearest_obj_idx
                # Continue interaction
                interaction_ids[i] = current_interaction_id
                object_names[i] = self.objects[current_object_idx]["semantic_label"]
            else:
                if in_interaction:
                    # End interaction
                    interaction_duration = times[i] - times[interaction_start]
                    if interaction_duration >= min_duration:
                        # Valid interaction, increment ID
                        current_interaction_id += 1
                    else:
                        # Too short, remove this interaction
                        for j in range(interaction_start, i):
                            interaction_ids[j] = current_interaction_id
                            object_names[j] = ""

                    in_interaction = False
                    current_object_idx = None

                # Walking state
                interaction_ids[i] = current_interaction_id
                object_names[i] = ""

        print(f"Detected {current_interaction_id + 1} interactions")
        return interaction_ids, object_names

    def add_action_labels(
        self, trajectory: pd.DataFrame, interaction_ids: np.ndarray
    ) -> List[str]:
        """
        Add action labels based on velocity and interactions

        Args:
            trajectory: DataFrame with t, x, y, z columns
            interaction_ids: Array of interaction IDs

        Returns:
            List of action labels
        """
        positions = trajectory[["x", "y", "z"]].values
        times = trajectory["t"].values

        # Compute velocity
        velocities = np.zeros(len(trajectory))
        for i in range(1, len(trajectory)):
            dt = times[i] - times[i-1]
            if dt > 0:
                velocities[i] = np.linalg.norm(positions[i] - positions[i-1]) / dt

        # Assign actions
        actions = []
        prev_interaction_id = interaction_ids[0]

        for i in range(len(trajectory)):
            # Check if in interaction
            if i > 0 and interaction_ids[i] == interaction_ids[i-1] and velocities[i] < 0.3:
                actions.append("interacting")
            elif velocities[i] > 0.5:
                actions.append("walking")
            elif velocities[i] > 0.1:
                actions.append("slow_walking")
            else:
                actions.append("standing")

        return actions

    def convert_to_lp2_format(
        self,
        trajectory: pd.DataFrame,
        interaction_threshold: float = 0.5,
        min_interaction_duration: float = 3.0,
    ) -> pd.DataFrame:
        """
        Convert trajectory to LP2 CSV format

        Args:
            trajectory: DataFrame with t, x, y, z columns
            interaction_threshold: Distance threshold for interactions
            min_interaction_duration: Minimum interaction duration

        Returns:
            DataFrame in LP2 format
        """
        # Infer room IDs
        room_ids = self.infer_room_ids(trajectory)

        # Infer interactions
        interaction_ids, object_names = self.infer_interactions(
            trajectory, interaction_threshold, min_interaction_duration
        )

        # Add action labels
        actions = self.add_action_labels(trajectory, interaction_ids)

        # Create LP2 DataFrame
        lp2_df = pd.DataFrame({
            "t": trajectory["t"].values,
            "x": trajectory["x"].values,
            "y": trajectory["y"].values,
            "z": trajectory["z"].values,
            "room_id": room_ids,
            "interaction_id": interaction_ids,
            "action": actions,
            "object": object_names,
        })

        return lp2_df


def main():
    parser = argparse.ArgumentParser(
        description="Convert trajectory data to LP2 CSV format"
    )
    parser.add_argument(
        "--input_trajectory",
        type=str,
        required=True,
        help="Path to input trajectory file"
    )
    parser.add_argument(
        "--input_format",
        type=str,
        choices=["xyz", "json", "csv"],
        default="xyz",
        help="Input format (default: xyz)"
    )
    parser.add_argument(
        "--scene_graph",
        type=str,
        required=True,
        help="Path to scene graph JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--trajectory_id",
        type=str,
        default="traj",
        help="Trajectory ID for output filename (default: traj)"
    )
    parser.add_argument(
        "--interaction_threshold",
        type=float,
        default=0.5,
        help="Distance threshold for interactions in meters (default: 0.5)"
    )
    parser.add_argument(
        "--min_interaction_duration",
        type=float,
        default=3.0,
        help="Minimum interaction duration in seconds (default: 3.0)"
    )

    args = parser.parse_args()

    # Load scene graph
    print(f"Loading scene graph: {args.scene_graph}")
    scene_graph = dsg.DynamicSceneGraph.load(args.scene_graph)

    # Create converter
    converter = TrajectoryConverter(scene_graph)

    # Load trajectory
    print(f"Loading trajectory: {args.input_trajectory}")
    trajectory = converter.load_trajectory(args.input_trajectory, args.input_format)

    # Convert to LP2 format
    print("\nConverting to LP2 format...")
    lp2_trajectory = converter.convert_to_lp2_format(
        trajectory,
        interaction_threshold=args.interaction_threshold,
        min_interaction_duration=args.min_interaction_duration,
    )

    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.trajectory_id}.csv"

    lp2_trajectory.to_csv(output_path)
    print(f"\nSaved LP2 trajectory: {output_path}")
    print(f"  Timesteps: {len(lp2_trajectory)}")
    print(f"  Duration: {lp2_trajectory['t'].max():.1f}s")
    print(f"  Interactions: {lp2_trajectory['interaction_id'].max() + 1}")


if __name__ == "__main__":
    main()
