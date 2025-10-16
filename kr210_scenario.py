import asyncio
import numpy as np
from pxr import Gf
from enum import Enum
from dataclasses import dataclass

# Isaac Sim Core Imports
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import XFormPrim, RigidPrim
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.world import World
import omni.physics.tensors as physics

# Motion Policy Imports
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
from omni.isaac.motion_generation.interface_config_loader import (
    get_supported_robot_policy_pairs,
    load_supported_motion_policy_config,
)

# **[FIXED IMPORT]** Using the fallback path for the gripper class
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper 

# **[LOCAL FIX]** Define the missing SurfaceGripperProperties class locally
@dataclass
class SurfaceGripperProperties:
    d6JointPath: str = ""
    parentPath: str = ""
    disableGravity: bool = True
    offset: object = None
    gripThreshold: float = 0.05
    forceLimit: float = 1.0e6
    torqueLimit: float = 1.0e6
    bendAngle: float = np.pi / 4
    stiffness: float = 1.0e4
    damping: float = 1.0e3
    retryClose: bool = True

# Local Imports
from .kr210_config import KR210Configuration

# --- Pick and Place State Machine ---
class PickPlaceState(Enum):
    IDLE = 0
    MOVE_TO_PICK = 1
    PICK = 2
    LIFT = 3
    MOVE_TO_PLACE = 4
    PLACE = 5
    RETRACT = 6
    COMPLETED = 7
    UNKNOWN = 8

class KR210Scenario:

    def __init__(self, kr210_config: KR210Configuration):
        self._rmpflow = None
        self._articulation_rmpflow = None
        self._stage = None
        self._articulation: Articulation = None
        self._target: XFormPrim = None
        self.pickup_objects_prims: list[XFormPrim] = []
        self.gripperTip: RigidPrim = None
        
        self.config = kr210_config

        self.pick_place_state = PickPlaceState.IDLE
        self.pickup_index = -1
        self.wait_frames = 0
        self.lifting_started = False
        
        self.target_lift_position = None
        self.target_lift_orientation = None

        # --- Gripper Properties Setup ---
        self.sgp = SurfaceGripperProperties() 
        self.sgp.d6JointPath = self.config.gripper_d6JointPath
        self.sgp.parentPath = self.config.gripper_parentPath
        self.sgp.disableGravity = True
        
        self.sgp.offset = physics.Transform()
        self.sgp.offset.p.x = 0.193000
        self.sgp.offset.p.z = -0.1001
        self.sgp.offset.r = [0.7071, 0, 0.7071, 0]
        self.sgp.gripThreshold = 0.05
        self.sgp.forceLimit = 1.0e6
        self.sgp.torqueLimit = 1.0e6
        self.sgp.bendAngle = np.pi / 4
        self.sgp.stiffness = 1.0e4
        self.sgp.damping = 1.0e3
        self.sgp.retryClose = True

        # **[CRITICAL FIX: DEFER INSTANTIATION]**
        self.surface_gripper = None 

        self.robot_endeffector = None

    def setup_assets(self):
        """Called POST-LOAD. All prims now exist and can be safely wrapped."""
        print("Setting up assets for scenario...")
        target_prim_path = "/World/target"
        self._stage = stage_utils.get_current_stage()
        world = World.instance()
        
        # 1. Articulation Setup
        if self._articulation is None:
            # **[CRITICAL FIX]** Only wrap the articulation if the prim is valid
            if prim_utils.is_prim_path_valid(self.config.robot_prim_path):
                self._articulation = Articulation(self.config.robot_prim_path)
                world.scene.add(self._articulation)
            else:
                print(f"ERROR: Robot prim {self.config.robot_prim_path} is not valid after load.")
                return # Stop if robot cannot be initialized
        
        # 2. Target Prim Setup
        if self._target is None:
            self._target = XFormPrim(target_prim_path, name="place_target")
            world.scene.add(self._target)
        
        # 3. **CRITICAL FIX: GRIPPER INSTANTIATION**
        if self.surface_gripper is None:
            print("Instantiating SurfaceGripper...")
            self.surface_gripper = SurfaceGripper(
                self.config.gripper_parentPath,      
                self.config.gripper_d6JointPath      
            ) 
            self.surface_gripper.initialize(self.sgp)
            print("SurfaceGripper successfully initialized.")


        # 4. Get End Effector and Pickup Prims
        self.robot_endeffector = self._stage.GetPrimAtPath(self.config.robot_endeffector_path)
        if not self.robot_endeffector.IsValid():
             print(f"Error: Robot endeffector prim not found at {self.config.robot_endeffector_path}")
             return

        self.pickup_objects_prims = []
        for pickup_object_path in self.config.pickup_objects:
            prim = self._stage.GetPrimAtPath(pickup_object_path)
            if prim.IsValid():
                pickup_name = pickup_object_path.split("/")[-1]
                pickup_prim = XFormPrim(pickup_object_path, name=pickup_name)
                self.pickup_objects_prims.append(pickup_prim)
                world.scene.add(pickup_prim)
            else:
                print(f"Warning: Pickup object not found at {pickup_object_path}")

        if self.gripperTip is None:
            self.gripperTip = RigidPrim(self.config.gripper_tip_path)
        
        print(f"Assets set up completed. Found {len(self.pickup_objects_prims)} pickup objects.")

    def setup(self):
        print("Setting up scenario RMPFlow...")

        # Initialize RMPFlow only after articulation exists
        rmp_config = load_supported_motion_policy_config("Kuka_KR210","RMPflow")
        self._rmpflow = RmpFlow(**rmp_config)
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation, self._rmpflow)
        
        self.reset()
        print("Scenario set up and reset completed.")

    def reset(self):
        print("Resetting scenario state...")
        self.pick_place_state = PickPlaceState.IDLE
        self.pickup_index = -1 
        self.wait_frames = 0
        self.lifting_started = False
        
        if self.surface_gripper is not None and self.surface_gripper.is_closed():
            self.surface_gripper.open()
            
        self.target_lift_position = None
        self.target_lift_orientation = None
        
        if self._rmpflow is not None:
             self._rmpflow.reset()
        
        print("Scenario reset completed.")
    
    def pick_cube(self):
        print("Picking up the cube initiated...")
        
        if self.pick_place_state is PickPlaceState.IDLE:
             self.pickup_index += 1
        
        if self.pickup_index >= len(self.pickup_objects_prims):
            print(f"Error: No more cubes to pick up! Index {self.pickup_index}, available cubes: {len(self.pickup_objects_prims)}")
            self.pickup_index = len(self.pickup_objects_prims) - 1
            return
        
        print(f"Picking up cube at index: {self.pickup_index}")
        self.pick_place_state = PickPlaceState.MOVE_TO_PICK

    def place_cube(self):
        print("Placing cube action (Execution driven by update loop).")
    
    def gripper_open(self):
        if self.surface_gripper is not None and self.surface_gripper.is_closed():
            self.surface_gripper.open()
            print("Gripper opened.")
        
    def gripper_close(self):
        if self.surface_gripper is not None and self.surface_gripper.is_closed() is False:
            self.surface_gripper.close()
            
    def update(self, step: float):
        if self.pick_place_state is PickPlaceState.IDLE or self._rmpflow is None:
            return

        if self.surface_gripper is not None:
             self.surface_gripper.update()

        # --- MOVE_TO_PICK Logic ---
        if self.pick_place_state is PickPlaceState.MOVE_TO_PICK:
            target_object = self.pickup_objects_prims[self.pickup_index]
            target_position, target_orientation = target_object.get_world_pose()
            target_position[0] -= self.config.pickup_offset 
            
            self._rmpflow.set_end_effector_target(target_position, target_orientation)

            self._rmpflow.update_world()
            self._rmpflow.set_robot_base_pose(*self._articulation.get_world_pose())
            action = self._articulation_rmpflow.get_next_articulation_action(step)
            self._articulation.apply_action(action)

            endeffector_pos_usd = self.robot_endeffector.GetAttribute("xformOp:translate").Get()
            ee_pos = np.array([endeffector_pos_usd[0], endeffector_pos_usd[1], endeffector_pos_usd[2]])
            target_pos_np = np.array(target_position)
            distance = np.linalg.norm(ee_pos - target_pos_np)
            
            if distance < 0.05:
                self.target_lift_position = target_position.copy()
                self.target_lift_orientation = target_orientation
                self.target_lift_position[0] += self.config.pickup_offset 
                self.target_lift_position_lift = self.target_lift_position.copy()
                self.target_lift_position_lift[2] += 0.5 
                
                self.pick_place_state = PickPlaceState.PICK
                self.wait_frames = 0

        # --- PICK Logic (Gripper close) ---
        elif self.pick_place_state is PickPlaceState.PICK:
            self.wait_frames += 1
            
            if self.wait_frames < 10:
                return
            
            if not self.surface_gripper.is_closed():
                self.gripper_close()
            
            self.wait_frames = 0
            self.pick_place_state = PickPlaceState.LIFT

        # --- LIFT Logic (Move up with cube) ---
        elif self.pick_place_state is PickPlaceState.LIFT:
            self._rmpflow.set_end_effector_target(self.target_lift_position_lift, self.target_lift_orientation)

            self._rmpflow.update_world() 
            self._rmpflow.set_robot_base_pose(*self._articulation.get_world_pose())
            action = self._articulation_rmpflow.get_next_articulation_action(step)
            self._articulation.apply_action(action)

            endeffector_pos_usd = self.robot_endeffector.GetAttribute("xformOp:translate").Get()
            ee_pos = np.array([endeffector_pos_usd[0], endeffector_pos_usd[1], endeffector_pos_usd[2]])
            distance = np.linalg.norm(ee_pos - self.target_lift_position_lift)

            if distance < 0.05:
                self.pick_place_state = PickPlaceState.MOVE_TO_PLACE

        # --- MOVE_TO_PLACE Logic (Move over to the drop-off area) ---
        elif self.pick_place_state is PickPlaceState.MOVE_TO_PLACE:
            
            place_position = np.array([0.5, -2.5, 2.0]) 
            self._rmpflow.set_end_effector_target(place_position, self.target_lift_orientation)

            self._rmpflow.update_world()
            self._rmpflow.set_robot_base_pose(*self._articulation.get_world_pose())
            action = self._articulation_rmpflow.get_next_articulation_action(step)
            self._articulation.apply_action(action)

            endeffector_pos_usd = self.robot_endeffector.GetAttribute("xformOp:translate").Get()
            ee_pos = np.array([endeffector_pos_usd[0], endeffector_pos_usd[1], endeffector_pos_usd[2]])
            distance = np.linalg.norm(ee_pos - place_position)

            if distance < 0.05:
                self.pick_place_state = PickPlaceState.PLACE

        # --- PLACE Logic (Lower the cube) ---
        elif self.pick_place_state is PickPlaceState.PLACE:
            
            final_place_position = np.array([0.5, -2.25, 0.95]) 
            self._rmpflow.set_end_effector_target(final_place_position, self.target_lift_orientation)

            self._rmpflow.update_world()
            self._rmpflow.set_robot_base_pose(*self._articulation.get_world_pose())
            action = self._articulation_rmpflow.get_next_articulation_action(step)
            self._articulation.apply_action(action)

            endeffector_pos_usd = self.robot_endeffector.GetAttribute("xformOp:translate").Get()
            ee_pos = np.array([endeffector_pos_usd[0], endeffector_pos_usd[1], endeffector_pos_usd[2]])
            distance = np.linalg.norm(ee_pos - final_place_position)

            if distance < 0.05:
                self.pick_place_state = PickPlaceState.RETRACT

        # --- RETRACT Logic (Open gripper, move to rest) ---
        elif self.pick_place_state is PickPlaceState.RETRACT:
            self.gripper_open()
            self.pick_place_state = PickPlaceState.COMPLETED

        # --- COMPLETED Logic (Move arm to a safe, neutral position) ---
        elif self.pick_place_state is PickPlaceState.COMPLETED:
            safe_retract_position = np.array([0.5, -1.0, 0.95]) 
            self._rmpflow.set_end_effector_target(safe_retract_position, self.target_lift_orientation)

            self._rmpflow.update_world()
            self._rmpflow.set_robot_base_pose(*self._articulation.get_world_pose())
            action = self._articulation_rmpflow.get_next_articulation_action(step)
            self._articulation.apply_action(action)