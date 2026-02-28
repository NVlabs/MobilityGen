import argparse
import pandas
from reader import Reader
import numpy as np
import tqdm
import PIL.Image
import io

parser = argparse.ArgumentParser()
parser.add_argument("recording_path")
parser.add_argument("output_path")
args = parser.parse_args()

reader = Reader(recording_path=args.recording_path)

index = 0


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


output: pandas.DataFrame = None

for index in tqdm.tqdm(range(len(reader))):

    data_dict = {}

    # Common data (saved as raw arrays)
    state_common = reader.read_state_dict_common(index=index)
    state_common.update(reader.read_state_dict_depth(index=index))
    state_common.update(reader.read_state_dict_segmentation(index=index))
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


output.to_parquet(args.output_path, engine="pyarrow")