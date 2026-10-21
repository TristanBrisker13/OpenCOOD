from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset

hypes = load_yaml('opencood/hypes_yaml/point_pillar_late_fusion.yaml')
print('root_dir:', hypes['root_dir'])
print('validate_dir:', hypes['validate_dir'])
print('batch_size:', hypes['train_params']['batch_size'])

for train in (True, False):
    ds = build_dataset(hypes, visualize=False, train=train)
    print('train' if train else 'val', 'samples:', len(ds))