# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class ScenarioTemplate:
    def __init__(self):
        pass

    def setup_scenario(self):
        pass

    def teardown_scenario(self):
        pass

    def update_scenario(self):
        pass


import numpy as np
from isaacsim.core.utils.types import ArticulationAction

"""
This scenario makes the KUKA robot perform a pick and place operation on the cube.
The robot will move through several phases:
1. Move to a position above the cube
2. Move down to grasp the cube
3. Lift the cube
4. Move to a placement position
5. Lower and release the cube
6. Return to home position
"""


class ExampleScenario(ScenarioTemplate):
    def __init__(self):
        self._object = None
        self._articulation = None
        self._running_scenario = False
        self._time = 0.0
        
        # Trajectory parameters
        self._phase = 0  # Current phase of pick and place
        self._phase_time = 0.0  # Time in current phase
        self._phase_duration = 2.0  # Duration for each phase (seconds)
        
        # Joint configurations for different poses
        self._home_position = None
        self._pre_grasp_position = None
        self._grasp_position = None
        self._lift_position = None
        self._place_position = None
        
        # Cube tracking
        self._cube_grasped = False
        self._initial_cube_position = None

    def setup_scenario(self, articulation, object_prim):
        self._articulation = articulation
        self._object = object_prim
        
        # Store initial cube position
        self._initial_cube_position = self._object.get_world_pose()[0]
        
        self._running_scenario = True
        self._phase = 0
        self._phase_time = 0.0
        self._cube_grasped = False
        
        # Define key joint configurations (these are example values - adjust based on your robot)
        # You may need to tune these positions for your specific KUKA model
        num_dof = articulation.num_dof
        
        # Home position - all joints at neutral
        self._home_position = np.zeros(num_dof)
        
        # Pre-grasp position - above the cube
        self._pre_grasp_position = np.array([0.0, -0.5, 0.5, 0.0, 0.5, 0.0][:num_dof])
        
        # Grasp position - at cube level
        self._grasp_position = np.array([0.0, -0.3, 0.3, 0.0, 0.8, 0.0][:num_dof])
        
        # Lift position - cube lifted
        self._lift_position = np.array([0.0, -0.5, 0.5, 0.0, 0.5, 0.0][:num_dof])
        
        # Place position - move to new location
        self._place_position = np.array([1.0, -0.3, 0.3, 0.0, 0.8, 0.0][:num_dof])
        
        # Start at home position
        articulation.set_joint_positions(self._home_position)
        
        print("Pick and Place Scenario Started")
        print("Phase 0: Moving to pre-grasp position")

    def teardown_scenario(self):
        self._time = 0.0
        self._phase_time = 0.0
        self._phase = 0
        self._object = None
        self._articulation = None
        self._running_scenario = False
        self._cube_grasped = False

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        
        self._time += step
        self._phase_time += step
        
        # Check if we should move to next phase
        if self._phase_time >= self._phase_duration:
            self._phase += 1
            self._phase_time = 0.0
            
            if self._phase == 1:
                print("Phase 1: Moving to grasp position")
            elif self._phase == 2:
                print("Phase 2: Grasping cube")
                self._cube_grasped = True
            elif self._phase == 3:
                print("Phase 3: Lifting cube")
            elif self._phase == 4:
                print("Phase 4: Moving to place position")
            elif self._phase == 5:
                print("Phase 5: Placing cube")
                self._cube_grasped = False
            elif self._phase == 6:
                print("Phase 6: Returning home")
            elif self._phase >= 7:
                print("Pick and Place Complete!")
                self._running_scenario = False
                return
        
        # Execute current phase
        self._execute_phase(step)
        
        # Update cube position if grasped
        if self._cube_grasped:
            self._update_cube_position()

    def _execute_phase(self, step):
        """Execute the current phase of the pick and place operation"""
        
        # Interpolation factor (0 to 1) within current phase
        t = min(self._phase_time / self._phase_duration, 1.0)
        
        # Get current joint positions
        current_positions = self._articulation.get_joint_positions()
        
        # Determine target position based on phase
        if self._phase == 0:
            # Move from home to pre-grasp
            target = self._interpolate_positions(self._home_position, self._pre_grasp_position, t)
        elif self._phase == 1:
            # Move from pre-grasp to grasp
            target = self._interpolate_positions(self._pre_grasp_position, self._grasp_position, t)
        elif self._phase == 2:
            # Stay at grasp position (grasping)
            target = self._grasp_position
        elif self._phase == 3:
            # Move from grasp to lift
            target = self._interpolate_positions(self._grasp_position, self._lift_position, t)
        elif self._phase == 4:
            # Move from lift to place
            target = self._interpolate_positions(self._lift_position, self._place_position, t)
        elif self._phase == 5:
            # Stay at place position (releasing)
            target = self._place_position
        elif self._phase == 6:
            # Move from place to home
            target = self._interpolate_positions(self._place_position, self._home_position, t)
        else:
            target = self._home_position
        
        # Apply smooth velocity control
        velocity = (target - current_positions) / max(step, 0.001)
        
        # Limit velocity
        max_velocity = 2.0  # rad/s
        velocity = np.clip(velocity, -max_velocity, max_velocity)
        
        # Apply action to robot
        action = ArticulationAction(
            joint_positions=target,
            joint_velocities=velocity
        )
        self._articulation.apply_action(action)

    def _interpolate_positions(self, start, end, t):
        """Smoothly interpolate between two joint configurations using cosine interpolation"""
        # Cosine interpolation for smoother motion
        smooth_t = (1 - np.cos(t * np.pi)) / 2
        return start + (end - start) * smooth_t

    def _update_cube_position(self):
        """Update cube position to follow the robot end-effector"""
        # Get the end-effector position (approximation using last joint)
        # In a real scenario, you would use forward kinematics
        
        # Simple approach: offset cube from robot base based on joint angles
        # This is a rough approximation - for accurate results, use proper FK
        joint_positions = self._articulation.get_joint_positions()
        
        # Approximate end-effector position based on joint configuration
        # You may need to adjust these calculations for your specific robot
        base_x = 0.0
        base_y = 0.0
        base_z = 0.0
        
        # Simple kinematic approximation (adjust link lengths as needed)
        reach = 1.5  # Approximate reach of robot
        x = base_x + reach * np.sin(joint_positions[0]) * np.cos(joint_positions[1])
        y = base_y + reach * np.cos(joint_positions[0]) * np.cos(joint_positions[1])
        z = base_z + reach * np.sin(joint_positions[1]) + 0.5
        
        # Set cube position
        self._object.set_world_pose(np.array([x, y, z]))