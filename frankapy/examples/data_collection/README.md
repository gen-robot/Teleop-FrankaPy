### setup environment
```bash
pip install h5py opencv-python transforms3d json_numpy requests
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu # cpu
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 # gpu
```

### driver install

**realsense**
```bash 
cd frankapy/drivers/realsense 
pip install -e .
```

**space_mouse**
You should follow the instructions in the [readme](../../drivers/space_mouse/README.md)
```bash 
cd frankapy/drivers/space_mouse
pip install -e .
```

### Use this command for data collection
```bash 
# Use ee pose impedance control in libfranka 
python -m examples.data_collection.data_collection --min_action_steps 50 --max_action_steps 1000 --instruction test --task_name bingwen  --episode_idx 1

# Use ee pose -> ik -> joint impedance control in libfranka
python -m examples.data_collection.data_collection_ik --min_action_steps 50 --max_action_steps 1000 --instruction test --task_name bingwen  --episode_idx 1 pos_scale 0.015 rot_scale 0.025 
```

### Note

The `control_frequency` should not set too high. We found a delay between camera return and control actions when `control_frequency=20`, it could be solved when `control_frequency` set to be 5.