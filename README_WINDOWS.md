# OpenCOOD on Windows with RTX 50-series GPUs

Extra setup steps needed **on top of the original OpenCOOD README** to train on Windows with a Blackwell GPU (tested on an RTX 5070 Ti, sm_120). Follow the original README for background, dataset download and model details. Use this file wherever it differs.

Tested with Python 3.10, PyTorch 2.11.0+cu128, spconv-cu126 2.3.8 and numpy 1.26.4.

> **Disclaimer:** This setup was put together and tested only on an **NVIDIA RTX 50-series (Blackwell) GPU**. The PyTorch, CUDA and spconv versions here were chosen because these cards require CUDA 12.8 or newer. If you have a different GPU, check which CUDA versions your card supports, then pick the PyTorch build (the `cuXXX` index URL in step 2) and the matching `spconv-cuXXX` / `cumm-cuXXX` and `nvidia-cuda-*` packages (step 4) to suit it. The Windows-specific fixes (the code change in step 6, the unpinned requirements, `--no-deps`, the dataset layout) apply regardless of GPU.

> **Note:** This repo will eventually be run on an NRP cluster. These instructions cover the local Windows setup only, and the cluster setup will be documented separately.

---

## Why the original instructions fail

| Problem | Cause |
|---|---|
| `CUDA capability sm_120 is not compatible` | RTX 50-series GPUs need PyTorch built for CUDA 12.8+, which needs Python ≥ 3.9. The original env targets an older Python. |
| `pip install -r requirements.txt` fails to build | Old pins (`numba==0.49.0`, `opencv-python~=4.5.1.48`, `scipy~=1.5.4`, …) have no wheels for Python 3.10, so pip tries to compile them from source. |
| `can't pickle ... Point2VoxelCPU` | Windows starts DataLoader workers with *spawn*, which pickles the dataset. The spconv voxel generator can't be pickled. |
| `DLL load failed while importing core_cc` | A conflicting `cumm` package, or missing CUDA runtime DLLs for cumm/spconv. |

---

## 1. Create the environment

Do **not** reuse a Python 3.7/3.8 env.

```powershell
conda create -n opencood2 python=3.10 -y
conda activate opencood2
```

## 2. Install PyTorch with CUDA 12.8

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "numpy<2"
```

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The version must end in `+cu128`, and the output should end with `True` and your GPU name. If it shows `+cpu` or `False`, something installed a CPU build from PyPI. Reinstall with `--force-reinstall` and the index URL above, then run `pip install "numpy<2"` again.

## 3. Install OpenCOOD's dependencies without the old pins

Strip the version pins from `requirements.txt` and install the latest compatible versions:

```powershell
(Get-Content requirements.txt) -replace '[=~<>!].*$','' | Where-Object { $_.Trim() -ne '' } | Set-Content requirements_py310.txt
```

Open `requirements_py310.txt` and **delete** any `torch`, `torchvision`, `spconv` or `cumm` lines, since those are installed separately. Then:

```powershell
pip install -r requirements_py310.txt "numpy<2"
```

If one package fails to build, remove its line and rerun. `open3d` is only needed for visualization.

## 4. Install spconv and the CUDA runtime DLLs

```powershell
pip install spconv-cu126
pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
```

Make sure there is exactly one `cumm` package:

```powershell
pip list | findstr /i "spconv cumm"
```

You should see only `cumm-cu126` and `spconv-cu126`. If a bare `cumm` is also listed, it conflicts, so remove everything and reinstall:

```powershell
pip uninstall -y cumm cumm-cu126 spconv-cu126
pip install spconv-cu126
```

Verify:

```powershell
python -c "import torch; import spconv.pytorch; print('ok')"
```

## 5. Install OpenCOOD itself

`setup.py` reads `requirements.txt` and would reinstall the old pins, so skip dependencies:

```powershell
pip install -e . --no-deps
python opencood/utils/setup.py build_ext --inplace
```

The second command needs the MSVC build tools (Visual Studio Build Tools, "Desktop development with C++" workload). If another compiler such as MinGW from Strawberry Perl is picked up instead, run it from an "x64 Native Tools Command Prompt for VS".

Ignore the `opencood 0.1.0 requires ...` warnings that pip prints after later installs.

## 6. Code change: make the voxel preprocessor picklable

Edit `opencood/data_utils/pre_processor/sp_voxel_preprocessor.py`.

First, save the constructor arguments on the **first line** of `__init__`. Note the leading underscore:

```python
    def __init__(self, preprocess_params, train):
        self._init_args = (preprocess_params, train)
        super(SpVoxelPreprocessor, self).__init__(preprocess_params,
                                                  train)
        # ... rest unchanged
```

Then add these two methods inside the class, after `__init__`:

```python
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('voxel_generator', None)  # spconv object can't be pickled
        return state

    def __setstate__(self, state):
        # rebuild everything, including voxel_generator, inside the worker
        self.__init__(*state['_init_args'])
```

This keeps multi-worker data loading working on Windows. Setting `num_workers=0` in `opencood/tools/train.py` also avoids the error, but data loading becomes much slower.

## 7. Dataset layout

OpenCOOD expects scenario folders **directly** inside each split:

```
opv2v_data_dumping/
├── train/
│   ├── 2021_08_16_22_26_54/
│   │   ├── 641/           <- one folder per vehicle (cav)
│   │   │   ├── 000069.pcd
│   │   │   ├── 000069.yaml
│   │   │   └── ...
│   │   └── data_protocol.yaml
│   └── ...
└── validate/
    └── ...
```

Unzipping often creates an extra level (`train/train/<scenario>`). If so, move the inner folders up one level, or point `root_dir` and `validate_dir` at the inner folder.

Symptoms of a wrong layout:

- `too many cavs` printed once per split, because the loader is treating scenarios as vehicles
- only 1 sample found
- training shows `0it` and crashes with `statistics.StatisticsError: mean requires at least one data point`

The `0it` crash also happens whenever a split has fewer samples than `batch_size`, because the loaders use `drop_last=True`.

## 8. Create the training subset

Training on the full OPV2V dataset isn't feasible on this setup, so train on a subset of evenly spaced scenarios. Copy them into a separate folder rather than deleting anything from the full dataset:

```powershell
function New-Subset($src, $dst, $n) {
    New-Item -ItemType Directory -Force $dst | Out-Null
    $all = Get-ChildItem $src -Directory | Sort-Object Name
    $step = [math]::Max(1, [math]::Floor($all.Count / $n))
    for ($k = 0; $k -lt $n; $k++) {
        Write-Host "Copying $($all[$k * $step].Name)..."
        Copy-Item $all[$k * $step].FullName -Destination $dst -Recurse
    }
}

New-Subset "opv2v_data_dumping\train"    "opv2v_subset\train"    5
New-Subset "opv2v_data_dumping\validate" "opv2v_subset\validate" 2
```

This uses 5 training and 2 validation scenarios. Change the numbers to adjust the subset size. Delete `opv2v_subset` before re-running with different numbers, so old scenarios don't mix in.

Point the config at the subset in `opencood/hypes_yaml/point_pillar_late_fusion.yaml`:

```yaml
root_dir: "opv2v_subset/train"
validate_dir: "opv2v_subset/validate"
train_params:
  batch_size: &batch_size 4
  epoches: 15
  eval_freq: 1
  save_freq: 1
```

Keep the subset and these settings the same for every run you want to compare. Results on a subset are useful for comparing runs against each other, but they will be lower than the published OPV2V numbers.

## 9. Check the dataset before training (optional)

This step isn't required, but it catches path and layout problems in seconds instead of after a failed training run. Save as `check_len.py` in the repo root:

```python
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset

hypes = load_yaml('opencood/hypes_yaml/point_pillar_late_fusion.yaml')
print('root_dir:', hypes['root_dir'])
print('validate_dir:', hypes['validate_dir'])
print('batch_size:', hypes['train_params']['batch_size'])

for train in (True, False):
    ds = build_dataset(hypes, visualize=False, train=train)
    print('train' if train else 'val', 'samples:', len(ds))
```

```powershell
python check_len.py
```

The paths should point at `opv2v_subset`, and both sample counts should be well above `batch_size`.

## 10. Train and evaluate

```powershell
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_late_fusion.yaml
```

A working run shows a progress bar with a real total (e.g. `135/135`), finite losses that decrease, a validation loss after each epoch, and `net_epochN.pth` checkpoints under `opencood/logs/<run_folder>/`.

To evaluate a trained model, which reports AP at IoU 0.3, 0.5 and 0.7:

```powershell
python opencood/tools/inference.py --model_dir opencood/logs/<run_folder> --fusion_method late
```

Add `--show_vis` or `--show_sequence` to view the detections in open3d.

To view loss curves:

```powershell
pip install tensorboard
tensorboard --logdir opencood/logs
```

---

## Harmless warnings

- `Detected call of lr_scheduler.step() before optimizer.step()`
- `The epoch parameter in scheduler.step() was not necessary`
- `opencood 0.1.0 requires ..., but you have ... which is incompatible` (from pip)
- `too many cavs`, when it appears only occasionally with a correct dataset layout. Some scenarios have more vehicles than `max_cav`.

## Troubleshooting

| Error | Fix |
|---|---|
| `Could not find a version that satisfies the requirement torch (from versions: none)` | Python version too old for cu128 wheels. Use Python 3.10. |
| `Could not find a version ... spconv-cu126` | Wrong env is active. Check that the prompt shows `(opencood2)`. |
| `CCompiler_spawn() got an unexpected keyword argument 'env'` during install | pip is building an old pinned package from source. Install from `requirements_py310.txt`, or use `pip install -e . --no-deps`. |
| `DLL load failed while importing core_cc` | Remove the bare `cumm` package and install the `nvidia-cuda-*-cu12` packages (step 4). |
| `Can't find cuda libraries, install cuda first` | `pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12`, and check that torch is a `+cu128` build. |
| `cannot pickle 'spconv...Point2VoxelCPU'` | Apply the code change in step 6. |
| `KeyError: '_init_args'` | The attribute in `__init__` is missing its leading underscore. |
| `AttributeError: module 'numpy' has no attribute 'float'` | numpy 2 was installed. Run `pip install "numpy<2"`. |
| Training shows `0it`, then `mean requires at least one data point` | Dataset path or layout is wrong, or there are fewer samples than `batch_size` (step 7). The optional check in step 9 shows the sample counts. |
| Training is slow (a few it/s) | Raise `num_workers` in `train.py`. |

## Recording a working environment

Once everything trains, save the exact versions:

```powershell
pip freeze > requirements_windows_working.txt
```
