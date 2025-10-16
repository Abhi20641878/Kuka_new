import numpy as np
import omni.timeline
import omni.ui as ui
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.ui_utils import get_style
from isaacsim.storage.native import get_assets_root_path
from omni.usd import StageEventType
from pxr import Sdf, UsdLux

from .kr210_scenario import KR210Scenario 
from .kr210_config import KR210Configuration


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
                self._load_btn = LoadButton(
                    "Load Button", "LOAD", setup_scene_fn=self._setup_scene, setup_post_load_fn=self._setup_scenario
                )
                self._load_btn.set_world_settings(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button", "RESET", pre_reset_fn=None, post_reset_fn=self._on_post_reset_btn
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

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

                self._pick_cube_button = ui.Button(
                    "Pick Cube",
                    clicked_fn=self._on_pick_cube_clicked)
                
                self._place_cube_button = ui.Button(
                    "Place Cube",
                    clicked_fn=self._on_place_cube_clicked)

    def _on_init(self):
        self._articulation = None
        self._cuboid = None
        
        config = KR210Configuration(
            pickup_objects=["/World/Default/Cubes/Pickup_A", "/World/Default/Cubes/Pickup_B", "/World/Default/Cubes/Pickup_C", "/World/Default/Cubes/Pickup_D"],
            pickup_offset=0.15,
            gripper_d6JointPath="/World/Default/kuka_kr210/link_6/SurfaceGripper",
            gripper_parentPath="/World/Default/kuka_kr210/link_6",
            gripper_tip_path="/World/Default/kuka_kr210/link_6",
            robot_endeffector_path="/World/Default/kuka_kr210/link_6",
            robot_prim_path="/World/Default/kuka_kr210"
        )
        self._kr210_scenario = KR210Scenario(config)

    def _add_light_to_stage(self):
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_poses(positions=np.array([[6.5, 0, 12]])) 

    def _setup_scene(self):
        """Called first. References USDs and creates World prims (like cubes)."""
        robot_prim_path = "/World/Default/kuka_kr210"

        create_new_stage()
        self._add_light_to_stage()
        add_reference_to_stage(robot_prim_path, robot_prim_path) 

        cube_paths = self._kr210_scenario.config.pickup_objects
        for i, cube_path in enumerate(cube_paths):
            FixedCuboid(
                cube_path,
                position=np.array([0.5 + i * 0.15, -0.5, 0.05]),
                size=0.05,
                color=np.array([255, 0, 0])
            )

    def _setup_scenario(self):
        """Called second (post-load). Stage is ready for Articulation/Gripper wrapping."""
        
        # **[CRITICAL FIX 1]** Force a single step to allow the USD prims to "cook" for physics.
        World.instance().step(render=False) 
        
        # **[CRITICAL FIX 2]** Setup assets now that the stage is ready.
        self._kr210_scenario.setup_assets() 
        self._reset_scenario()
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _reset_scenario(self):
        self._kr210_scenario.setup() 

    def _on_post_reset_btn(self):
        self._kr210_scenario.setup_assets()
        self._reset_scenario()
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float):
        self._kr210_scenario.update(step)

    def _on_run_scenario_a_text(self):
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        self._timeline.pause()

    def _on_pick_cube_clicked(self):
        if self._kr210_scenario:
            self._kr210_scenario.pick_cube()

    def _on_place_cube_clicked(self):
        if self._kr210_scenario:
            self._kr210_scenario.place_cube()

    def _reset_extension(self):
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False