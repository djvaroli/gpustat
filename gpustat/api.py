from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import json
from typing import *
from pathlib import Path

from blessed import Terminal

from gpustat import __version__
from gpustat.core import GPUStatCollection


def get_gpustat_json(debug: bool = False) -> dict:
    """
    Get the GPU query results in json format
    """

    try:
        gpu_stats = GPUStatCollection.new_query()
    except Exception as e:
        sys.stderr.write('Error on querying NVIDIA devices.'
                         ' Use set debug to True for details\n')
        if debug:
            try:
                import traceback
                traceback.print_exc(file=sys.stderr)
            except Exception:
                # NVMLError can't be processed by traceback:
                #   https://bugs.python.org/issue28603
                # as a workaround, simply re-throw the exception
                raise e

    return gpu_stats.jsonify()


def gpu_stat_to_file(filename: str, **kwargs) -> None:
    """
    Saves gpustats to a json file
    """
    json_gpu_stats = get_gpustat_json(**kwargs)
    json_gpu_stats['query_time'] = json_gpu_stats['query_time'].isoformat()
    path_to_file = Path(filename)

    mode = "w"
    if path_to_file.exists():
        mode = "a"
    
    gpu_stats_str = json.dumps(json_gpu_stats)
    with open(path_to_file, mode=mode) as f:
        f.write(f"{gpu_stats_str}\n")


def load_gpu_stats_from_file(filename: str) -> List[dict]:
    """
    Loads gpu stats from a json file
    """

    lines = (l for l in open(filename))
    gpu_stats = []
    for line in lines:
        gpu_stats.append(json.loads(line))
    
    return gpu_stats