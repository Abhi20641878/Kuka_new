# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import omni.timeline
import omni.ui as ui
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux

from .scenario import ExampleScenario


class UIBuilder:
    def __init__(self):
        self.frames = []
        self.wrapped_ui_elements = []
        self._timeline = omni.timeline.get_timeline_interface()
        self._on_init()

    def on_menu_callback(self):
        pass

    def on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        pass

    def on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            self._reset_extension()

    def cleanup(self):
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)

        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # Create custom load button
                with ui.HStack():
                    ui.Button(
                        "LOAD",
                        clicked_fn=self._on_load_world,
                        height=40,
                        style=get_style()
                    )

                with ui.HStack():
                    ui.Button(
                        "RESET",
                        clicked_fn=self._on_reset_world,
                        height=40,
                        style=get_style(),
                        enabled=False,
                        identifier="reset_button"
                    )

        run_scenario_frame = CollapsableFrame("Run Scenario")

        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

    def _on_init(self):
        self._articulation = None
        self._cuboid = None
        self._scenario = ExampleScenario()
        self._world = None

    def _add_light_to_stage(self):
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_poses(np.array([[6.5, 0, 12]]))

    def _on_load_world(self):
        """Custom load function that properly manages World creation"""
        print("\n" + "="*50)
        print("Loading KUKA Pick and Place Scenario...")
        print("="*50)
        
        # Step 1: Create new stage and load assets
        create_new_stage()
        self._add_light_to_stage()
        
        # Load the KUKA robot
        robot_prim_path = "/kr210_l150"
        path_to_robot_usd = "D:/Ext/kuka/data/Collected_kr210_l150/kr210_l150.usd"
        
        print(f"Loading robot from: {path_to_robot_usd}")
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        
        # Create cuboid
        print("Creating cuboid...")
        self._cuboid = FixedCuboid(
            "/Scenario/cuboid", 
            position=np.array([0.6, 0.3, 0.3]),
            size=0.05, 
            color=np.array([255, 0, 0])
        )

        # Create articulation wrapper
        print("Creating articulation wrapper...")
        self._articulation = SingleArticulation(robot_prim_path)

        # Step 2: Create World with physics settings
        print("Creating World...")
        self._world = World(physics_dt=1/60.0, rendering_dt=1/60.0)
        
        # Step 3: Add objects to world
        print("Adding objects to World scene...")
        self._world.scene.add(self._articulation)
        self._world.scene.add(self._cuboid)
        
        # Step 4: Initialize world (this calls reset internally)
        print("Initializing World...")
        self._world.reset()
        
        # Step 5: Setup scenario
        print("Setting up scenario...")
        self._reset_scenario()
        
        # Enable UI
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._enable_reset_button(True)
        
        print("\n" + "="*50)
        print("✓ KUKA Pick and Place Ready!")
        print("✓ Click RUN to start the scenario")
        print("="*50 + "\n")

    def _on_reset_world(self):
        """Reset the world and scenario"""
        if self._world is None:
            print("World not loaded yet!")
            return
            
        print("\nResetting world...")
        self._world.reset()
        self._reset_scenario()
        
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        print("World reset complete!\n")

    def _enable_reset_button(self, enabled):
        """Helper to enable/disable reset button"""
        # Find and enable the reset button
        import omni.ui as ui
        window = ui.Workspace.get_window("KUKA Robot Extension")  # Adjust name if different
        if window:
            # Button identifier set in build_ui
            pass

    def _reset_scenario(self):
        if self._articulation is None or self._cuboid is None:
            print("Warning: Cannot reset scenario - objects not loaded")
            return
            
        self._scenario.teardown_scenario()
        self._scenario.setup_scenario(self._articulation, self._cuboid)

    def _update_scenario(self, step: float):
        self._scenario.update_scenario(step)

    def _on_run_scenario_a_text(self):
        print("\n>>> Starting Pick and Place <<<\n")
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        print("\n>>> Stopping Scenario <<<\n")
        self._timeline.pause()

    def _reset_extension(self):
        if self._world:
            self._world.clear_instance()
        self._on_init()
        if hasattr(self, '_scenario_state_btn'):
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False