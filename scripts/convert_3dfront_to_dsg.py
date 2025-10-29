#!/usr/bin/env python3
"""
Convert 3D FRONT dataset to Spark-DSG format for LP2

This script converts 3D FRONT scene JSON files to the Spark-DSG format required by LP2.
It creates:
1. Scene graph JSON (Spark-DSG format with OBJECTS, PLACES, ROOMS layers)
2. Room labels JSON (mapping room IDs to semantic labels)

Usage:
    python scripts/convert_3dfront_to_dsg.py \
        --input_json path/to/3dfront_scene.json \
        --output_dir data/3dfront_scene/ \
        --scene_id scene_0001

Author: Claude Code
Date: 2025-10-29
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import spark_dsg as dsg


class ThreeDFrontToDSGConverter:
    """Convert 3D FRONT scenes to Spark-DSG format"""

    def __init__(self):
        """Initialize the converter"""
        # Layer IDs following LP2 convention
        self.OBJECTS_LAYER = 2
        self.PLACES_LAYER = 3
        self.ROOMS_LAYER = 4

        # Mapping from 3D FRONT room types to semantic labels
        self.room_type_mapping = {
            "LivingRoom": "living room",
            "MasterRoom": "bedroom",
            "SecondRoom": "bedroom",
            "Kitchen": "kitchen",
            "Bathroom": "bathroom",
            "DiningRoom": "dining room",
            "ChildRoom": "child room",
            "StudyRoom": "study room",
            "Bedroom": "bedroom",
            "LivingDiningRoom": "living dining room",
            "Balcony": "balcony",
            "Corridor": "corridor",
            "Other": "other",
        }

        # Mapping from 3D FUTURE category to semantic labels
        self.furniture_category_mapping = {
            "sofa": "sofa",
            "chair": "chair",
            "table": "table",
            "bed": "bed",
            "desk": "desk",
            "cabinet": "cabinet",
            "shelf": "shelf",
            "tv_stand": "tv stand",
            "nightstand": "nightstand",
            "dining_table": "dining table",
            "dining_chair": "dining chair",
            "coffee_table": "coffee table",
            "wardrobe": "wardrobe",
            "bookshelf": "bookshelf",
            "tv": "tv",
            "stool": "stool",
            "bench": "bench",
            "armchair": "armchair",
            "dresser": "dresser",
            "plant": "plant",
            "lamp": "lamp",
            "ceiling_lamp": "ceiling lamp",
            "pendant_lamp": "pendant lamp",
        }

    def load_3dfront_scene(self, json_path: str) -> Dict:
        """Load 3D FRONT scene JSON file"""
        with open(json_path, 'r') as f:
            scene_data = json.load(f)
        return scene_data

    def extract_furniture_info(self, furniture_item: Dict, room_bbox: Dict) -> Dict:
        """
        Extract furniture information from 3D FRONT format

        Args:
            furniture_item: Furniture item from 3D FRONT JSON
            room_bbox: Room bounding box for coordinate context

        Returns:
            Dictionary with furniture info (category, position, bbox, etc.)
        """
        # Extract position (center of the furniture)
        # 3D FRONT uses "pos" field with [x, y, z] coordinates
        position = furniture_item.get("pos", [0, 0, 0])

        # Extract category/semantic label
        # In 3D FRONT, this is typically in "category" or "type" field
        category = furniture_item.get("category", "unknown")

        # Clean category name (remove prefixes like "Children_", etc.)
        category_clean = category.lower().replace("children_", "").replace("_", " ")
        semantic_label = self.furniture_category_mapping.get(category_clean, category_clean)

        # Extract bounding box if available
        bbox = furniture_item.get("bbox", None)

        # Extract rotation
        rotation = furniture_item.get("rot", [0, 0, 0, 1])  # quaternion [x, y, z, w]

        # Extract scale
        scale = furniture_item.get("scale", [1, 1, 1])

        return {
            "jid": furniture_item.get("jid", ""),  # 3D FUTURE model ID
            "uid": furniture_item.get("uid", ""),  # unique instance ID
            "category": category,
            "semantic_label": semantic_label,
            "position": position,
            "bbox": bbox,
            "rotation": rotation,
            "scale": scale,
        }

    def compute_room_center(self, room_data: Dict) -> List[float]:
        """
        Compute room center from bounding box

        Args:
            room_data: Room data from 3D FRONT JSON

        Returns:
            [x, y, z] center position
        """
        bbox = room_data.get("bbox", {})

        # 3D FRONT bbox format: {"min": [x, y, z], "max": [x, y, z]}
        if "min" in bbox and "max" in bbox:
            min_pos = bbox["min"]
            max_pos = bbox["max"]
            center = [
                (min_pos[0] + max_pos[0]) / 2,
                (min_pos[1] + max_pos[1]) / 2,
                (min_pos[2] + max_pos[2]) / 2,
            ]
        else:
            center = [0, 0, 0]

        return center

    def create_place_positions_for_room(
        self, room_center: List[float], furniture_positions: List[List[float]],
        grid_size: float = 1.0
    ) -> List[List[float]]:
        """
        Create place node positions for a room
        Places are spatial locations that link rooms to objects

        Args:
            room_center: Center position of the room
            furniture_positions: List of furniture positions in the room
            grid_size: Grid spacing for place nodes

        Returns:
            List of [x, y, z] positions for place nodes
        """
        # Strategy: Create place nodes near each furniture item
        # and some additional places for navigation

        place_positions = []

        # Add places near each furniture
        for furn_pos in furniture_positions:
            # Add place at furniture location
            place_positions.append(furn_pos.copy())

            # Optionally: Add places around furniture (4 cardinal directions)
            # for pos_offset in [[grid_size, 0, 0], [-grid_size, 0, 0],
            #                   [0, grid_size, 0], [0, -grid_size, 0]]:
            #     offset_pos = [furn_pos[i] + pos_offset[i] for i in range(3)]
            #     place_positions.append(offset_pos)

        # Add a place at room center
        if room_center not in place_positions:
            place_positions.append(room_center)

        return place_positions

    def convert_to_dsg(
        self, scene_data: Dict, scene_id: str
    ) -> Tuple[dsg.DynamicSceneGraph, Dict[int, str]]:
        """
        Convert 3D FRONT scene to Spark-DSG format

        Args:
            scene_data: 3D FRONT scene JSON data
            scene_id: Scene identifier

        Returns:
            Tuple of (DynamicSceneGraph, room_labels_dict)
        """
        # Create a new Dynamic Scene Graph
        scene_graph = dsg.DynamicSceneGraph()

        # Room labels dictionary: {room_id: semantic_label}
        room_labels = {}

        # Parse scene data
        # Note: 3D FRONT structure may vary, adjust field names as needed
        scene_info = scene_data.get("scene", scene_data)
        rooms = scene_info.get("room", [])
        furniture_list = scene_info.get("furniture", [])

        # Track IDs
        room_id_counter = 0
        place_id_counter = 0
        object_id_counter = 0

        # Maps for linking
        room_to_places = {}  # room_id -> list of place_ids
        place_to_objects = {}  # place_id -> list of object_ids

        # Process each room
        for room_idx, room_data in enumerate(rooms):
            room_type = room_data.get("type", "Other")
            semantic_label = self.room_type_mapping.get(room_type, room_type.lower())

            # Create room node
            room_node_id = room_id_counter
            room_labels[room_node_id] = semantic_label

            # Compute room center
            room_center = self.compute_room_center(room_data)

            # Add room node to scene graph
            room_attrs = dsg.NodeAttributes()
            room_attrs.position = np.array(room_center, dtype=np.float64)
            room_attrs.name = semantic_label

            # Create node symbol for room (R = Rooms layer)
            room_symbol = dsg.NodeSymbol('R', room_node_id)
            scene_graph.add_node(self.ROOMS_LAYER, room_symbol.value, room_attrs)

            # Get furniture in this room
            room_furniture_ids = room_data.get("children", [])
            room_furniture = [f for f in furniture_list if f.get("uid") in room_furniture_ids]

            # Extract furniture info
            furniture_info_list = []
            furniture_positions = []
            for furn in room_furniture:
                furn_info = self.extract_furniture_info(furn, room_data.get("bbox", {}))
                furniture_info_list.append(furn_info)
                furniture_positions.append(furn_info["position"])

            # Create place nodes for this room
            place_positions = self.create_place_positions_for_room(
                room_center, furniture_positions
            )

            room_place_ids = []
            for place_pos in place_positions:
                place_node_id = place_id_counter
                room_place_ids.append(place_node_id)

                # Add place node
                place_attrs = dsg.NodeAttributes()
                place_attrs.position = np.array(place_pos, dtype=np.float64)
                place_attrs.name = f"place_{place_node_id}"

                place_symbol = dsg.NodeSymbol('P', place_node_id)
                scene_graph.add_node(self.PLACES_LAYER, place_symbol.value, place_attrs)

                # Add interlayer edge: Room -> Place
                scene_graph.insert_interlayer_edge(room_symbol.value, place_symbol.value)

                place_id_counter += 1

            room_to_places[room_node_id] = room_place_ids

            # Create object nodes for furniture
            for furn_info in furniture_info_list:
                object_node_id = object_id_counter

                # Add object node
                object_attrs = dsg.NodeAttributes()
                object_attrs.position = np.array(furn_info["position"], dtype=np.float64)
                object_attrs.name = furn_info["semantic_label"]

                object_symbol = dsg.NodeSymbol('O', object_node_id)
                scene_graph.add_node(self.OBJECTS_LAYER, object_symbol.value, object_attrs)

                # Find closest place node to this object
                closest_place_id = None
                min_distance = float('inf')
                for place_id in room_place_ids:
                    place_symbol_temp = dsg.NodeSymbol('P', place_id)
                    place_node = scene_graph.get_node(place_symbol_temp.value)
                    distance = np.linalg.norm(
                        np.array(furn_info["position"]) - place_node.attributes.position
                    )
                    if distance < min_distance:
                        min_distance = distance
                        closest_place_id = place_id

                # Add interlayer edge: Place -> Object
                if closest_place_id is not None:
                    closest_place_symbol = dsg.NodeSymbol('P', closest_place_id)
                    scene_graph.insert_interlayer_edge(
                        closest_place_symbol.value, object_symbol.value
                    )

                    if closest_place_id not in place_to_objects:
                        place_to_objects[closest_place_id] = []
                    place_to_objects[closest_place_id].append(object_node_id)

                object_id_counter += 1

            room_id_counter += 1

        return scene_graph, room_labels

    def save_scene_graph(self, scene_graph: dsg.DynamicSceneGraph, output_path: str):
        """Save scene graph to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene_graph.save(str(output_path))
        print(f"Saved scene graph to: {output_path}")

    def save_room_labels(self, room_labels: Dict[int, str], output_path: str):
        """Save room labels to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(room_labels, f, indent=2)
        print(f"Saved room labels to: {output_path}")

    def convert_scene(
        self, input_json_path: str, output_dir: str, scene_id: str
    ):
        """
        Main conversion function

        Args:
            input_json_path: Path to 3D FRONT scene JSON file
            output_dir: Output directory for converted files
            scene_id: Scene identifier (e.g., "scene_0001")
        """
        print(f"Converting 3D FRONT scene: {input_json_path}")
        print(f"Scene ID: {scene_id}")

        # Load 3D FRONT scene
        scene_data = self.load_3dfront_scene(input_json_path)
        print(f"Loaded scene with {len(scene_data.get('scene', {}).get('room', []))} rooms")

        # Convert to DSG
        scene_graph, room_labels = self.convert_to_dsg(scene_data, scene_id)

        # Print statistics
        print("\nConversion Statistics:")
        print(f"  Rooms layer: {len(list(scene_graph.get_layer(self.ROOMS_LAYER).nodes))} nodes")
        print(f"  Places layer: {len(list(scene_graph.get_layer(self.PLACES_LAYER).nodes))} nodes")
        print(f"  Objects layer: {len(list(scene_graph.get_layer(self.OBJECTS_LAYER).nodes))} nodes")
        print(f"  Interlayer edges: {len(list(scene_graph.interlayer_edges))}")

        # Save outputs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_graph_path = output_dir / f"{scene_id}_scene_graph.json"
        room_labels_path = output_dir / f"{scene_id}_room_labels.json"

        self.save_scene_graph(scene_graph, str(scene_graph_path))
        self.save_room_labels(room_labels, str(room_labels_path))

        print(f"\nConversion complete!")
        return scene_graph_path, room_labels_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D FRONT dataset to Spark-DSG format for LP2"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to 3D FRONT scene JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted files"
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default="3dfront_scene",
        help="Scene identifier (default: 3dfront_scene)"
    )

    args = parser.parse_args()

    # Create converter and run conversion
    converter = ThreeDFrontToDSGConverter()
    converter.convert_scene(
        input_json_path=args.input_json,
        output_dir=args.output_dir,
        scene_id=args.scene_id
    )


if __name__ == "__main__":
    main()
