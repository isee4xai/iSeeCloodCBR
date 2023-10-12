# Reuse functions
"""
Instruction on including new reuse scripts.
------------------------------------------
A new custom reuse operation should have file name that begins with underscore (e.g., _<script_name>.py) and added to the 'custom_reuse_scripts' folder.
The function which takes the input (request body) and returns the output of the reuse operation should be name 'reuse'.
To use the function, add 'reuse_type' property to the input with value as '_<script_name>'.
"""

import importlib


def reuse_cases(data):
  # get reuse type
  reuse_type = data.get('reuse_type', None)
  reuse_feature = data.get('reuse_feature', None)
  if reuse_type is None or reuse_feature is None:
    return None
  elif reuse_type.startswith('_'):  # import custom module for use operation
    module_name = 'custom_reuse_scripts.' + reuse_type
    reuse_module = importlib.import_module(module_name)
    # execute and return
    if reuse_feature is 'transform':
      return reuse_module.transform_adapt(data)
    elif reuse_feature is 'applicability':
      return reuse_module.applicability(data)
    elif reuse_feature is 'substitute':
      return reuse_module.substitute(data)
  else:  # generic reuse operation
    # logic for any generic reuse operations below
    return None
