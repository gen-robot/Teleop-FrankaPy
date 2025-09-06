### Examples

```bash
# For package installation.
pip install -e .
# You can try the command below.

# link camera, read the number of camera, save some of the images of camera.
python frankapy/drivers/realsense/python/realsense_wrapper/realsense_d435.py

# realtime display all the camera in a window, tuning the camera parameter.
python frankapy/drivers/realsense/python/realsense_wrapper/camera_display.py
```

<!-- 
### Not Use 25.5.17
Bingwen comment, this readme can be discarded.

Installation:
```
pip install git+https://github.com/facebookresearch/fairo.git@main#subdirectory=perception/realsense_driver
```

Optional conda environment
```
conda create -n eyehandcal polymetis librealsense -c fair-robotics
pip install git+https://github.com/facebookresearch/fairo.git@main#subdirectory=perception/realsense_driver
```

Usage:
```py
from realsense_wrapper import RealsenseAPI

rs = RealsenseAPI()

num_cameras = rs.get_num_cameras()
intrinsics = rs.get_intrinsics()
imgs = rs.get_images()
``` -->