

# SpaceMouse Driver Installation Guide (spacenavd & libspnav)

This guide will help you install and configure the open-source drivers `spacenavd` and `libspnav` for SpaceMouse on Ubuntu, and test the device.
[Note] Installing the official 3dconnexion driver may conflict with this tutorial. The official Linux driver is too old (2014), so this method is recommended. Reference: [CSDN article](https://blog.csdn.net/qq_40081208/article/details/137675822)


## 1. Install spacenavd

### 1.1 Install dependencies

Run the following commands in the terminal to install required dependencies:

```bash
sudo apt install libxext-dev libxrender-dev libxmu-dev libxmuu-dev
sudo apt-get install libxtst*
sudo apt-get install libx11*
sudo apt-get install libxi-dev
```


### 1.2 Download and install spacenavd

```bash
git clone https://github.com/FreeSpacenav/spacenavd.git
cd spacenavd
./configure
make
sudo make install
```


### 1.3 Set spacenavd to start on boot

```bash
sudo ./setup_init # auto start when open the computer
sudo /etc/init.d/spacenavd start # start now 
```

> **Note: You may see error messages like the following during the process. They do not affect usage and can be ignored.**
>
> ```
> cat: /etc/inittab: No such file or directory
> default runlevel detection failed.
> ```

---


## 2. Install libspnav

### 2.1 Download and install libspnav

```bash
git clone https://github.com/FreeSpacenav/libspnav.git
cd libspnav
./configure
make
sudo make install
```

---


## 3. Verify driver installation

### 3.1 Connect the device

Connect the SpaceMouse to your computer via USB.

### 3.2 Run the test program

```bash
cd libspnav/examples/simple
make
./simple_af_unix
```


### 3.3 Example of normal output

If the device is connected correctly, you will see output similar to the following in the terminal, indicating the driver is installed successfully:

```
spacenav AF_UNIX protocol version: 1
Device: 3Dconnexion SpaceMouse Wireless
Path: /dev/input/event6
Buttons: 2
Axes: 6
got motion event: t(0, 0, 0) r(-15, 0, 0)
got motion event: t(0, -5, 0) r(-32, 0, 0)
...
got motion event: t(-143, 0, 105) r(-142, -31, -131)
```



## 4. Configure hid (SpaceMouse device permissions)

1.  **Enter the udev rules directory:**
    ```bash
    sudo vim /etc/udev/rules.d/99-spacemouse.rules
    ```

2.  **Add the following content to the new `99-spacemouse.rules` file:**
    ```udev
    SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0666"
    ```
    **Important:**
    Please replace `256f` (Vendor ID) and `c635` (Product ID) with the actual Vendor ID and Product ID of your SpaceMouse device. You can find these IDs by running the `lsusb` command. You should see something like `Bus 003 Device 003: ID 256f:c635 3Dconnexion SpaceMouse Compact`. If you cannot find `3Dconnexion SpaceMouse Compac`, your version of the `usb.ids` file is too old; you should use `sudo update -usbids` to update the file. For example, if your device shows `ID 256f:c635`, the rule should be:
    ```udev
    SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0666"
    ```

3.  **[Optional, I don't use] Apply the udev rules:**
    ```bash
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    ```

4.  **Replug your SpaceMouse device** or restart your computer to ensure the new permission rules are applied.



## 5. Run python script test

### 5.1 In your python script
```python
space_mouse = SpaceMouse(vendor_id = 0x256f, product_id = 0xc635) # replace it with your own id.
# in class
vendor_id=0x256f, # on Bingwen's computer
product_id=0xc635, # on Bingwen's computer
```

### 5.2 Install required packages and run
Install hidapi, pynput
```bash
pip install hidapi pynput # instead of hid, or conda install in conda_ros # sudo apt install libudev0, if you can't find libudev.so.0.
python space_mouse.py
```

---


## References

- [spacenavd GitHub repository](https://github.com/FreeSpacenav/spacenavd)
- [libspnav GitHub repository](https://github.com/FreeSpacenav/libspnav)

---
