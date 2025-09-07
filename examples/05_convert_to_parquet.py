import argparse
import pandas
from reader import Reader
import numpy as np
import tqdm
import PIL.Image
import io

import os
import subprocess
import glob
import argparse


def numpy_array_to_flattened_columns(key: str, value: np.ndarray):
    columns = {
        f"{key}": value.flatten()
    }
    # add shape if ndim > 1
    if value.ndim > 1:
        columns[f"{key}.shape"] = tuple(value.shape)
    return columns


def numpy_array_to_jpg_columns(key: str, value: np.ndarray):
    image = PIL.Image.fromarray(value)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    columns = {
        key: buffer.getvalue()
    }
    return columns


if "MOBILITY_GEN_DATA" in os.environ:
    DATA_DIR = os.environ['MOBILITY_GEN_DATA']
else:
    DATA_DIR = os.path.expanduser("~/MobilityGenData")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)


    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = os.path.join(DATA_DIR, "replays")

    if args.output_dir is None:
        args.output_dir = os.path.join(DATA_DIR, "parquet")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    input_recordings = glob.glob(os.path.join(args.input_dir, "*"))

    processed_count = 0

    for input_recording_path in input_recordings:
        processed_count += 1
        print(f"Processing {processed_count} / {len(input_recordings)}")

        recording_name = os.path.basename(input_recording_path)
        output_path = os.path.join(args.output_dir, recording_name + ".pqt")

        reader = Reader(recording_path=input_recording_path)

        index = 0


        output: pandas.DataFrame = None

        for index in tqdm.tqdm(range(len(reader))):

            data_dict = {}

            # Common data (saved as raw arrays)
            state_common = reader.read_state_dict_common(index=index)
            state_common.update(reader.read_state_dict_depth(index=index))
            state_common.update(reader.read_state_dict_segmentation(index=index))
            # state_common.update(reader.read_state_dict_depth(index=index))
            # TODO: handle normals
            
            for k, v in state_common.items():
                if isinstance(v, np.ndarray):
                    columns = numpy_array_to_flattened_columns(k, v)
                else:
                    columns = {k: v}
                data_dict.update(columns)

            # RGB data (saved as jpg)
            state_rgb = reader.read_state_dict_rgb(index=index)
            for k, v in state_rgb.items():
                if isinstance(v, np.ndarray):
                    columns = numpy_array_to_jpg_columns(k, v)
                else:
                    columns = {k: v}
                data_dict.update(columns)

            
            # use first frame to initialize
            if output is None:
                output = pandas.DataFrame(columns=data_dict.keys())

            output.loc[index] = data_dict


        output.to_parquet(output_path, engine="pyarrow")