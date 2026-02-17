# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gradio as gr
import argparse
import numpy as np
import matplotlib.pyplot as plt
from reader import Reader
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="~/MobilityGenData/replays")
args = parser.parse_args()

directory = os.path.expanduser(args.input_dir)
recording_names = [os.path.basename(path) for path in glob.glob(os.path.join(directory, "*"))]

class Context:
    def __init__(self, recording_name: str):
        self.reader = Reader(os.path.join(directory, recording_name))
        self.occupancy_map = self.reader.read_occupancy_map()
        self._path_xy = None
    
    def get_path_xy(self):
        if self._path_xy is not None:
            return self._path_xy
        points_world = []

        # Get path points
        for index in range(len(self.reader)):
            state_dict = self.reader.read_state_dict_common(index)
            position = state_dict['robot.position']
            points_world.append(position[0:2])
        points_world = np.array(points_world)
        points_image = self.occupancy_map.world_to_pixel_numpy(points_world)
        self._path_xy = image = points_image
        return self._path_xy
    
    def get_robot_xy(self, index: int):
        state_dict = self.reader.read_state_dict_common(index)
        pos_world = state_dict['robot.position']
        pos_image = self.occupancy_map.world_to_pixel_numpy(np.array([pos_world[0:2]]))
        return pos_image

    def get_map_plot(self, index: int):
        path_xy = self.get_path_xy()
        robot_xy = self.get_robot_xy(index)
        fig, ax = plt.subplots()
        ax.imshow(self.occupancy_map.ros_image().convert("RGB"))
        ax.plot(path_xy[0, 0], path_xy[0, 1], 'go')
        ax.plot(path_xy[:, 0], path_xy[:, 1], '--', linewidth=3)
        ax.plot(path_xy[-1, 0], path_xy[-1, 1], 'ro')
        ax.plot(robot_xy[0, 0], robot_xy[0, 1], 'g*', markersize=15)
        ax.axis('off')
        return fig

    def update(self, index: int):
        state_dict = self.reader.read_state_dict(index=int(index))
        return [
            self.get_map_plot(index),
            state_dict['robot.front_camera.left.rgb_image'],
            np.clip(1.0 / state_dict['robot.front_camera.left.depth_image'], 0., 1.),
            state_dict['robot.front_camera.right.rgb_image'],
            np.clip(1.0 / state_dict['robot.front_camera.right.depth_image'], 0., 1.)
        ]


context = None

def update_recording(name: str):
    global context
    context = Context(name)
    main = context.update(index=0)
    slider = gr.Slider(value=0, minimum=0, maximum=len(context.reader) - 1, step=1)
    return main + [slider]


def update_step(index: int):
    return context.update(index)

    
with gr.Blocks(title="MobilityGen - Data Explorer", fill_height=True) as demo:
    with gr.Column(scale=2):
        with gr.Row(equal_height=False):
            recording_selector = gr.Radio(label=f"{os.path.realpath(args.input_dir)}", choices=recording_names, scale=2)
            with gr.Column(scale=7):
                with gr.Row(equal_height=True):
                    map_plot = gr.Plot(scale=2, show_label=False)
                    with gr.Column(scale=1):
                        left_rgb = gr.Image(show_label=False)
                        left_depth = gr.Image(show_label=False)
                    with gr.Column(scale=1):
                        right_rgb = gr.Image(show_label=False)
                        right_depth = gr.Image(show_label=False)
                with gr.Row():
                    slider = gr.Slider(label="Timestep", minimum=0, maximum=0, step=1)

                update_widgets = [map_plot, left_rgb, left_depth, right_rgb, right_depth]

                recording_selector.change(update_recording, recording_selector, update_widgets + [slider])
                slider.change(update_step, slider, update_widgets, show_progress=False)

if __name__ == "__main__":
    demo.launch()