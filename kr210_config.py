from dataclasses import dataclass
from typing import List

@dataclass
class KR210Configuration:
    # Define prim paths of object that needs to be picked up by the robot.
    pickup_objects: List[str]
    
    # Offset to avoid collision with the cube when picking it up
    pickup_offset: float

    # Gripper properties
    gripper_d6JointPath: str
    gripper_parentPath: str
    gripper_tip_path: str

    # Robot properties
    robot_endeffector_path: str
    # Default path for the Kuka robot in the USD stage
    robot_prim_path: str = "/World/Default/kuka_kr210"