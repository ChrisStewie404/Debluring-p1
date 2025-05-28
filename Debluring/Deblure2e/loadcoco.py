from ultralytics.utils import ASSETS,yaml_load
from ultralytics.utils.checks import check_requirements,check_yaml

classes = yaml_load(check_yaml("coco8.yaml"))["names"]
print(classes)