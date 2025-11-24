## Kinematics Modular Installation

### Assets

```bash
# To download the panda robot assets
python frankapy/scripts/check_and_download.py --robot_name "panda" 
```

### ManiSkill Kinematics
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu # cpu
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 # gpu
pip install pytorch_kinematics==0.7.5 sapien==3.0.0.b1
```

### Pyroki Kinematics
```bash
pip install git+https://github.com/chungmin99/pyroki.git@70b30a5
pip install "numpy<2.0" opencv-python
```
