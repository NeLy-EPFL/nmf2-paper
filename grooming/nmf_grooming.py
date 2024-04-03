from flygym import Fly, Camera
from flygym.state import KinematicPose
import flygym.preprogrammed as preprogrammed


import pickle
from pathlib import Path
import numpy as np

# List of all the joints that are actuated during grooming
all_groom_dofs = (
    [f"joint_{dof}" for dof in ["Head", "Head_yaw", "Head_roll"]]  # Head joints
    + [
        f"joint_{side}F{dof}"
        for side in "LR"
        for dof in [
            "Coxa",
            "Coxa_roll",
            "Coxa_yaw",
            "Femur",
            "Femur_roll",
            "Tibia",
            "Tarsus1",
        ]
    ]  # Front leg joints
    + [
        f"joint_{side}{dof}{angle}"
        for side in "LR"
        for dof in ["Pedicel"]
        for angle in ["", "_yaw"]
    ]  # Antennae joints
)

# List of alL the bodies that might be colliding during groomming

groom_self_collision = [
    f"{side}{app}"
    for side in "LR"
    for app in [
        "FTibia",
        "FTarsus1",
        "FTarsus2",
        "FTarsus3",
        "FTarsus4",
        "FTarsus5",
        "Arista",
        "Funiculus",
        "Pedicel",
        "Eye",
    ]
]

class GroomingFly(Fly):
    def __init__(self, xml_variant, self_collisions=groom_self_collision, actuator_kp=10.0, contact_sensor_placements=preprogrammed.all_tarsi_links):
        super().__init__(
            xml_variant=xml_variant,
            self_collisions=self_collisions,
            actuator_kp=actuator_kp,
            contact_sensor_placements=contact_sensor_placements,
            init_pose=KinematicPose.from_yaml("./data/pose_groom.yaml"),
            actuated_joints=all_groom_dofs,
            floor_collisions = "none",
        )

    def _set_joints_stiffness_and_damping(self):
        super()._set_joints_stiffness_and_damping()
        for joint in self.model.find_all("joint"):
            if any([app in joint.name for app in ["Pedicel", "Arista", "Funiculus"]]):
                joint.stiffness = 1e-3
                joint.damping = 1e-3

        return None

    def _set_actuators_gain(self):
        for actuator in self.actuators:
            if "Arista" in actuator.name:
                kp = 1e-6
            elif "Pedicel" in actuator.name or "Funiculus" in actuator.name:
                kp = 0.2
            else:
                kp = self.actuator_kp
            actuator.kp = kp
        return None
    
    def _define_self_contacts(self, self_collisions_geoms):
        # Only add relevant collisions:
        # - No collisions between the two antennas
        # - No collisions between segments in the same leg or antenna

        self_contact_pairs = []
        self_contact_pairs_names = []

        for geom1_name in self_collisions_geoms:
            for geom2_name in self_collisions_geoms:
                body1 = self.model.find("geom", geom1_name).parent
                body2 = self.model.find("geom", geom2_name).parent
                simple_body1_name = body1.name.split("_")[0]
                simple_body2_name = body2.name.split("_")[0]

                body1_children = self.get_real_childrens(body1)
                body2_children = self.get_real_childrens(body2)

                body1_parent = self._get_real_parent(body1)
                body2_parent = self._get_real_parent(body2)

                geom1_is_antenna = any(
                    [
                        app in geom1_name
                        for app in ["Pedicel", "Arista", "Funiculus", "Eye"]
                    ]
                )
                geom2_is_antenna = any(
                    [
                        app in geom2_name
                        for app in ["Pedicel", "Arista", "Funiculus", "Eye"]
                    ]
                )
                is_same_side = geom1_name[0] == geom2_name[0]

                if not (
                    body1.name == body2.name
                    or simple_body1_name in body2_children
                    or simple_body2_name in body1_children
                    or simple_body1_name == body2_parent
                    or simple_body2_name == body1_parent
                    or geom1_is_antenna
                    and geom2_is_antenna  # both on antenna
                    or is_same_side
                    and (
                        not geom1_is_antenna and not geom2_is_antenna
                    )  # on the legs and same side
                ):
                    contact_pair = self.model.contact.add(
                        "pair",
                        name=f"{geom1_name}_{geom2_name}",
                        geom1=geom1_name,
                        geom2=geom2_name,
                        solref=self.sim_params.contact_solref,
                        solimp=self.sim_params.contact_solimp,
                        margin=0.0,  # change margin to avoid penetration
                    )
                    self_contact_pairs.append(contact_pair)
                    self_contact_pairs_names.append(f"{geom1_name}_{geom2_name}")

        return self_contact_pairs, self_contact_pairs_names


class GroomingCamera(Camera):
    def __init__(self, fly, **kwargs):
        super().__init__(fly, **kwargs)
        self.update_camera_pos = False
        self._zoom_camera(fly)

    
    def _zoom_camera(self, fly):
        if self.camera_id == f"{fly.name}/camera_front":
            self.cam_offset -= [
                8-2.7,
                0.0,
                0.0,
            ]
        else: 
            raise ValueError(f"Camera {self.camera_id} not recognized")

    """def _add_force_sensors(self):
        super()._add_force_sensors()
        self._add_touch_sensors()

    def _add_touch_sensors(self):
        touch_sensors = []
        for body_name in self.touch_sensor_locations:
            body = self.model.find("body", body_name)
            body_child_names = self.get_real_childrens(body)
            if body_child_names:
                body_child_name = body_child_names[0]
                body_child = self.model.find("body", body_child_name)
                next_body_pos = body_child.pos
                if np.sum(np.abs(next_body_pos)) < 1e-3:
                    next_body_pos = [0.0, 0.0, -0.2]
            elif "Arista" in body_name:
                if body_name[0] == "L":
                    next_body_pos = [0.0, -0.2, 0.0]
                else:
                    next_body_pos = [0.0, 0.2, 0.0]
            elif "Funiculus" in body_name:
                next_body_pos = [0.0, 0.0, -0.2]
            elif "Tarsus" in body_name:
                next_body_pos = [0.0, 0.0, -0.2]
            elif "Eye" in body_name:
                pass
            else:
                raise ValueError(f"Body {body_name} has no children")
            if "Eye" in body_name:
                site = body.add(
                    "site",
                    type="sphere",
                    name=f"{body_name}_touchsite",
                    pos=next_body_pos,
                    size=[1.0, 0.0, 0.0],
                    rgba=[0.0, 0.0, 0.0, 0.0],
                )
            else:
                quat = body.quat
                site = body.add(
                    "site",
                    type="capsule",
                    quat=quat,
                    fromto=np.hstack((np.zeros(3), next_body_pos)),
                    name=f"{body_name}_touchsite",
                    pos=[0, 0, 0],
                    size=[0.1, 0.0, 0.0],
                    rgba=[0.0, 0.0, 0.0, 0.0],
                )
            touch_sensor = self.model.sensor.add(
                "touch", name=f"touch_{body.name}", site=site.name
            )
            touch_sensors.append(touch_sensor)
        self.touch_sensors = touch_sensors

    def get_observation(self):
        obs = super().get_observation()
        touch_data = self.physics.bind(self.touch_sensors).sensordata
        obs["touch_sensors"] = touch_data.copy()
        return obs"""