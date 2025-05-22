### setup environment
```bash
pip install h5py opencv-python transforms3d json_numpy requests
# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
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
```bash 
cd frankapy/drivers/space_mouse
pip install -e .
```

### Use this command for data collection
```bash 
python -m examples.data_collection.data_collection --min_action_steps 200 --max_action_steps 1000 --instruction test --task_name bingwen # --episode_idx 1
```