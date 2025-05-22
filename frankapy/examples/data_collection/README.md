### setup environment
```bash
pip install h5py opencv-python transforms3d json_numpy requests
# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu # cpu
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 # gpu
```


```bash 
python -m examples.data_collection.data_collection --task_name bingwen --instruction test
```