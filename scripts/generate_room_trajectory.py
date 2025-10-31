#!/usr/bin/env python3
"""
Generate realistic single-room trajectories from 3D FRONT room data

This script generates trajectories that:
1. Visit multiple objects in a room naturally
2. Follow realistic paths considering furniture placement
3. Create both past trajectory (for LP2 input) and full trajectory (for evaluation)

Strategy:
- Semantic-based visiting order (functional relationships)
- Spatial-based path planning (avoid furniture)
- Distance-based interaction duration
- Natural human-like motion patterns

Usage:
    python scripts/generate_room_trajectory.py \
        --room_json path/to/3dfront_room.json \
        --output_dir data/trajectories/room_001/ \
        --num_trajectories 10

Author: Claude Code
Date: 2025-10-31
"""

import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull
import random


class RoomTrajectoryGenerator:
    """Generate realistic trajectories within a single room"""

    def __init__(
        self,
        room_data: Dict,
        furniture_data: List[Dict],
        walking_speed: float = 1.0,
        interaction_duration_range: Tuple[float, float] = (5.0, 20.0),
        furniture_clearance: float = 0.5,  # meters around furniture to avoid
    ):
        """
        Initialize the generator

        Args:
            room_data: 3D FRONT room dictionary
            furniture_data: List of furniture dictionaries in this room
            walking_speed: Average walking speed (m/s)
            interaction_duration_range: (min, max) interaction duration (seconds)
            furniture_clearance: Safety distance around furniture (meters)
        """
        self.room_data = room_data
        self.furniture_data = furniture_data
        self.walking_speed = walking_speed
        self.interaction_duration_range = interaction_duration_range
        self.furniture_clearance = furniture_clearance

        # Extract room info
        self.room_type = room_data.get("type", "Unknown")
        self.room_bbox = room_data.get("bbox", {})

        # Furniture semantic relationships for visiting order
        self.semantic_relationships = self._build_semantic_relationships()

        # Compute accessible positions
        self.accessible_positions = self._compute_accessible_positions()

    def _build_semantic_relationships(self) -> Dict[str, List[str]]:
        """
        Build semantic relationships between furniture types
        Objects that are often used together are grouped
        """
        relationships = {
            # Living room
            "sofa": ["coffee_table", "tv_stand", "side_table", "armchair"],
            "coffee_table": ["sofa", "armchair"],
            "tv_stand": ["sofa", "tv"],
            "tv": ["tv_stand", "sofa"],

            # Bedroom
            "bed": ["nightstand", "dresser", "wardrobe"],
            "nightstand": ["bed", "lamp"],
            "dresser": ["bed", "wardrobe"],
            "wardrobe": ["dresser", "bed"],

            # Kitchen
            "stove": ["counter", "refrigerator", "sink"],
            "sink": ["counter", "stove", "dishwasher"],
            "refrigerator": ["counter", "dining_table"],
            "counter": ["stove", "sink", "refrigerator"],
            "dining_table": ["dining_chair", "refrigerator"],
            "dining_chair": ["dining_table"],

            # Bathroom
            "toilet": ["sink", "bathtub"],
            "sink": ["toilet", "mirror"],
            "bathtub": ["sink", "toilet"],

            # Study
            "desk": ["chair", "bookshelf", "lamp"],
            "chair": ["desk"],
            "bookshelf": ["desk", "chair"],
        }
        return relationships

    def _compute_accessible_positions(self) -> List[np.ndarray]:
        """
        Compute accessible positions in the room (avoiding furniture)
        Uses a simple grid-based approach
        """
        bbox = self.room_bbox
        if "min" not in bbox or "max" not in bbox:
            # Default room size
            return [np.array([0, 0, 0])]

        min_pos = np.array(bbox["min"])
        max_pos = np.array(bbox["max"])

        # Create grid
        grid_resolution = 0.5  # 50cm grid
        x_range = np.arange(min_pos[0], max_pos[0], grid_resolution)
        y_range = np.arange(min_pos[1], max_pos[1], grid_resolution)

        accessible = []
        for x in x_range:
            for y in y_range:
                pos = np.array([x, y, 0.0])  # Assume z=0 for floor level

                # Check if position is clear of furniture
                is_clear = True
                for furn in self.furniture_data:
                    furn_pos = np.array(furn.get("pos", [0, 0, 0]))
                    furn_pos[2] = 0  # Project to floor

                    dist = np.linalg.norm(pos - furn_pos)
                    if dist < self.furniture_clearance:
                        is_clear = False
                        break

                if is_clear:
                    accessible.append(pos)

        return accessible if accessible else [np.array([0, 0, 0])]

    def _get_furniture_position(self, furniture: Dict) -> np.ndarray:
        """Get furniture interaction position (front of furniture)"""
        pos = np.array(furniture.get("pos", [0, 0, 0]))

        # Get rotation to determine "front"
        rot = furniture.get("rot", [0, 0, 0, 1])  # quaternion

        # Simple heuristic: offset in front direction
        # For now, just add small offset in y direction
        front_offset = np.array([0, 0.3, 0])  # 30cm in front

        return pos + front_offset

    def _compute_furniture_category(self, furniture: Dict) -> str:
        """Normalize furniture category"""
        category = furniture.get("category", "unknown").lower()
        # Remove prefixes
        category = category.replace("children_", "").replace("_", " ")
        return category

    def generate_visiting_sequence(
        self,
        num_interactions: int = 5,
        strategy: str = "semantic"
    ) -> List[Dict]:
        """
        Generate a sequence of furniture to visit

        Args:
            num_interactions: Number of furniture pieces to interact with
            strategy: 'semantic' (follow relationships), 'spatial' (nearest first),
                     'random', or 'circular' (go around room)

        Returns:
            List of furniture dictionaries in visiting order
        """
        if len(self.furniture_data) == 0:
            return []

        num_interactions = min(num_interactions, len(self.furniture_data))

        if strategy == "semantic":
            return self._generate_semantic_sequence(num_interactions)
        elif strategy == "spatial":
            return self._generate_spatial_sequence(num_interactions)
        elif strategy == "circular":
            return self._generate_circular_sequence(num_interactions)
        else:  # random
            return random.sample(self.furniture_data, num_interactions)

    def _generate_semantic_sequence(self, num_interactions: int) -> List[Dict]:
        """Generate sequence based on semantic relationships"""
        visited = []
        available = self.furniture_data.copy()

        # Start with random furniture
        current = random.choice(available)
        visited.append(current)
        available.remove(current)

        while len(visited) < num_interactions and available:
            current_category = self._compute_furniture_category(current)

            # Find related furniture
            related_categories = self.semantic_relationships.get(current_category, [])

            candidates = [
                f for f in available
                if self._compute_furniture_category(f) in related_categories
            ]

            if not candidates:
                # No semantic match, use closest
                candidates = available

            # Among candidates, choose closest
            current_pos = np.array(current.get("pos", [0, 0, 0]))
            candidates.sort(
                key=lambda f: np.linalg.norm(
                    np.array(f.get("pos", [0, 0, 0])) - current_pos
                )
            )

            next_furn = candidates[0]
            visited.append(next_furn)
            available.remove(next_furn)
            current = next_furn

        return visited

    def _generate_spatial_sequence(self, num_interactions: int) -> List[Dict]:
        """Generate sequence by visiting nearest neighbors"""
        visited = []
        available = self.furniture_data.copy()

        # Start with random furniture
        current = random.choice(available)
        visited.append(current)
        available.remove(current)

        while len(visited) < num_interactions and available:
            current_pos = np.array(current.get("pos", [0, 0, 0]))

            # Find nearest
            available.sort(
                key=lambda f: np.linalg.norm(
                    np.array(f.get("pos", [0, 0, 0])) - current_pos
                )
            )

            next_furn = available[0]
            visited.append(next_furn)
            available.remove(next_furn)
            current = next_furn

        return visited

    def _generate_circular_sequence(self, num_interactions: int) -> List[Dict]:
        """Generate sequence by going around room (clockwise/counterclockwise)"""
        if not self.furniture_data:
            return []

        # Compute room center
        bbox = self.room_bbox
        if "min" in bbox and "max" in bbox:
            center = (np.array(bbox["min"]) + np.array(bbox["max"])) / 2
        else:
            center = np.mean([np.array(f.get("pos", [0, 0, 0]))
                             for f in self.furniture_data], axis=0)

        # Sort furniture by angle from center
        def angle_from_center(furniture):
            pos = np.array(furniture.get("pos", [0, 0, 0]))
            rel_pos = pos - center
            return np.arctan2(rel_pos[1], rel_pos[0])

        sorted_furniture = sorted(self.furniture_data, key=angle_from_center)

        # Take evenly spaced furniture
        if len(sorted_furniture) <= num_interactions:
            return sorted_furniture
        else:
            step = len(sorted_furniture) // num_interactions
            return [sorted_furniture[i * step] for i in range(num_interactions)]

    def generate_smooth_path(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        num_waypoints: int = 20
    ) -> np.ndarray:
        """
        Generate smooth path between two positions, avoiding furniture

        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            num_waypoints: Number of points along path

        Returns:
            Array of shape (num_waypoints, 3)
        """
        # Simple approach: cubic spline with midpoint offset
        direction = end_pos - start_pos
        distance = np.linalg.norm(direction)

        if distance < 0.1:
            return np.array([start_pos] * num_waypoints)

        # Add 1-2 control points for natural curve
        num_control = 2
        control_points = [start_pos]

        for i in range(num_control):
            t = (i + 1) / (num_control + 1)
            mid = start_pos + t * direction

            # Add perpendicular offset for natural path
            perpendicular = np.array([-direction[1], direction[0], 0])
            perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-8)

            offset = perpendicular * random.uniform(-0.3, 0.3) * distance
            control_points.append(mid + offset)

        control_points.append(end_pos)
        control_points = np.array(control_points)

        # Cubic spline interpolation
        t_control = np.linspace(0, 1, len(control_points))
        t_fine = np.linspace(0, 1, num_waypoints)

        path = np.zeros((num_waypoints, 3))
        for dim in range(3):
            cs = CubicSpline(t_control, control_points[:, dim])
            path[:, dim] = cs(t_fine)

        return path

    def generate_trajectory(
        self,
        num_interactions: int = 5,
        total_duration: float = 120.0,
        strategy: str = "semantic",
        split_at_interaction: int = 2,  # Split past/future after this interaction
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a complete trajectory with train/test split

        Args:
            num_interactions: Number of furniture interactions
            total_duration: Total trajectory duration (seconds)
            strategy: Visiting order strategy
            split_at_interaction: Split into past/future after this interaction

        Returns:
            Tuple of (past_trajectory, full_trajectory)
        """
        # Generate visiting sequence
        visiting_sequence = self.generate_visiting_sequence(num_interactions, strategy)

        if not visiting_sequence:
            # Empty trajectory
            empty_df = pd.DataFrame(columns=["t", "x", "y", "z", "room_id",
                                             "interaction_id", "action", "object"])
            return empty_df, empty_df

        # Generate trajectory points
        trajectory_data = []
        current_time = 0.0
        interaction_id = 0

        for furn_idx, furniture in enumerate(visiting_sequence):
            # Interaction position
            furn_pos = self._get_furniture_position(furniture)
            furn_category = self._compute_furniture_category(furniture)

            # Interaction duration
            interaction_duration = random.uniform(*self.interaction_duration_range)

            # Add interaction points
            timestep = 0.1  # 10 Hz
            num_steps = int(interaction_duration / timestep)

            for step in range(num_steps):
                t = current_time + step * timestep
                if t > total_duration:
                    break

                # Small random motion during interaction
                noise = np.random.normal(0, 0.03, 3)  # 3cm std dev
                pos = furn_pos + noise

                trajectory_data.append({
                    "t": t,
                    "x": pos[0],
                    "y": pos[1],
                    "z": pos[2],
                    "room_id": 0,  # Single room
                    "interaction_id": interaction_id,
                    "action": f"interacting_{furn_category}",
                    "object": furn_category,
                })

            current_time += interaction_duration
            interaction_id += 1

            if current_time > total_duration or furn_idx == len(visiting_sequence) - 1:
                break

            # Walking to next furniture
            next_furniture = visiting_sequence[furn_idx + 1]
            next_pos = self._get_furniture_position(next_furniture)

            path = self.generate_smooth_path(furn_pos, next_pos)
            path_distance = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
            walking_duration = path_distance / self.walking_speed

            num_walking_steps = int(walking_duration / timestep)
            for step in range(num_walking_steps):
                t = current_time + step * timestep
                if t > total_duration:
                    break

                # Interpolate along path
                progress = step / max(num_walking_steps - 1, 1)
                path_idx = int(progress * (len(path) - 1))
                pos = path[path_idx]

                trajectory_data.append({
                    "t": t,
                    "x": pos[0],
                    "y": pos[1],
                    "z": pos[2],
                    "room_id": 0,
                    "interaction_id": interaction_id,
                    "action": "walking",
                    "object": "",
                })

            current_time += walking_duration

        # Convert to DataFrame
        full_trajectory = pd.DataFrame(trajectory_data)

        # Split into past/future
        # Find the end of split_at_interaction
        split_rows = full_trajectory[
            full_trajectory["interaction_id"] == split_at_interaction
        ]

        if len(split_rows) > 0:
            # Add 2 seconds after last interaction
            split_time = split_rows.iloc[-1]["t"] + 2.0
            past_trajectory = full_trajectory[full_trajectory["t"] <= split_time].copy()
        else:
            # Fallback: split at 20% of duration
            split_time = total_duration * 0.2
            past_trajectory = full_trajectory[full_trajectory["t"] <= split_time].copy()

        return past_trajectory, full_trajectory


def load_3dfront_room(json_path: str) -> Tuple[Dict, List[Dict]]:
    """
    Load 3D FRONT room JSON and extract room and furniture data

    Args:
        json_path: Path to 3D FRONT scene JSON

    Returns:
        Tuple of (room_data, furniture_list)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract room and furniture
    # Adjust these field names based on your actual 3D FRONT JSON structure
    scene = data.get("scene", data)

    # If multiple rooms, take the first one
    rooms = scene.get("room", [])
    if not rooms:
        raise ValueError("No rooms found in JSON")

    room = rooms[0]  # Take first room

    # Get furniture in this room
    all_furniture = scene.get("furniture", [])
    room_furniture_ids = room.get("children", [])

    furniture_in_room = [
        f for f in all_furniture
        if f.get("uid") in room_furniture_ids
    ]

    print(f"Loaded room type: {room.get('type')}")
    print(f"Found {len(furniture_in_room)} furniture pieces")

    return room, furniture_in_room


def main():
    parser = argparse.ArgumentParser(
        description="Generate single-room trajectories from 3D FRONT data"
    )
    parser.add_argument(
        "--room_json",
        type=str,
        required=True,
        help="Path to 3D FRONT room JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for trajectories"
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=10,
        help="Number of trajectories to generate (default: 10)"
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=5,
        help="Number of furniture interactions per trajectory (default: 5)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Total trajectory duration in seconds (default: 120.0)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["semantic", "spatial", "circular", "random"],
        default="semantic",
        help="Visiting order strategy (default: semantic)"
    )
    parser.add_argument(
        "--split_at_interaction",
        type=int,
        default=2,
        help="Split past/future after this interaction number (default: 2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load room data
    print(f"Loading 3D FRONT room: {args.room_json}")
    room_data, furniture_data = load_3dfront_room(args.room_json)

    # Create generator
    generator = RoomTrajectoryGenerator(room_data, furniture_data)

    # Create output directories
    output_dir = Path(args.output_dir)
    past_dir = output_dir / "past"
    full_dir = output_dir / "full"
    past_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {args.num_trajectories} trajectories...")
    print(f"  Strategy: {args.strategy}")
    print(f"  Interactions per trajectory: {args.num_interactions}")
    print(f"  Duration: {args.duration}s")
    print(f"  Split at interaction: {args.split_at_interaction}")

    # Generate trajectories
    for i in range(args.num_trajectories):
        past_traj, full_traj = generator.generate_trajectory(
            num_interactions=args.num_interactions,
            total_duration=args.duration,
            strategy=args.strategy,
            split_at_interaction=args.split_at_interaction,
        )

        # Save trajectories
        past_path = past_dir / f"traj_{i:04d}.csv"
        full_path = full_dir / f"traj_{i:04d}.csv"

        past_traj.to_csv(past_path)
        full_traj.to_csv(full_path)

        print(f"Generated trajectory {i+1}/{args.num_trajectories}: "
              f"past={len(past_traj)} steps, full={len(full_traj)} steps")

    print(f"\n✓ Generation complete!")
    print(f"  Past trajectories: {past_dir}/")
    print(f"  Full trajectories: {full_dir}/")
    print(f"\nUse past trajectories as input to LP2")
    print(f"Use full trajectories for evaluation")


if __name__ == "__main__":
    main()
