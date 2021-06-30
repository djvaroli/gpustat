from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from blessed import Terminal

from gpustat import __version__
from .core import GPUStatCollection


def get_gpustat(debug=False):
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
