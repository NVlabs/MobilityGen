import numpy as np

from mobility_gen_path_planner import generate_paths, compress_path

start = (0, 0)
freespace = np.ones((40, 40), dtype=bool)

output = generate_paths(start, freespace)

path = output.unroll_path(end=(20, 39))

cpath, ckeep = compress_path(path)

print(path)
print(cpath)
print(ckeep)