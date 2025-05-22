
# SpaceMouse 驱动安装指南（spacenavd & libspnav）

本指南将帮助你在 Ubuntu 系统中安装和配置 SpaceMouse 的开源驱动 `spacenavd` 与 `libspnav`，并进行设备测试。 
[注意] 直接安装 3dconnexion 驱动可能会与此教程冲突。原官网linux驱动过于老旧且为2014版本，故采用此种安装方法。本文参考 [CSDN网站](https://blog.csdn.net/qq_40081208/article/details/137675822)

## 1. 安装 spacenavd

### 1.1 安装依赖库

在终端中执行以下命令安装所需依赖：

```bash
sudo apt install libxext-dev libxrender-dev libxmu-dev libxmuu-dev
sudo apt-get install libxtst*
sudo apt-get install libx11*
sudo apt-get install libxi-dev
```

### 1.2 下载和安装 spacenavd

```bash
git clone https://github.com/FreeSpacenav/spacenavd.git
cd spacenavd
./configure
make
sudo make install
```

### 1.3 设置 spacenavd 开机启动

```bash
sudo ./setup_init
sudo /etc/init.d/spacenavd start
```

> **注意：过程中可能会看到类似如下报错信息，不影响使用，可以忽略。**
>
> ```
> cat: /etc/inittab: 没有那个文件或目录
> default runlevel detection failed.
> ```

---

## 2. 安装 libspnav

### 2.1 下载和安装 libspnav

```bash
git clone https://github.com/FreeSpacenav/libspnav.git
cd libspnav
./configure
make
sudo make install
```

---

## 3. 验证驱动安装是否成功

### 3.1 连接设备

将 SpaceMouse 通过 USB 连接到电脑。

### 3.2 运行测试程序

```bash
cd libspnav/examples/simple
make
./simple_af_unix
```

### 3.3 正常输出示例

运行后，若设备连接正确，终端中会输出如下类似信息，表示驱动安装成功：

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


## 4. 配置 hid (SpaceMouse 设备权限)

1.  **进入 udev 规则目录：**
    ```bash
    sudo vim /etc/udev/rules.d/99-spacemouse.rules
    ```

2.  **在新建的 `99-spacemouse.rules` 文件中添加以下内容：**
    ```udev
    SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c62e", MODE="0666"
    ```
    **重要提示：**
    请将上述规则中的 `256f` (Vendor ID) 和 `c62e` (Product ID) **替换为您自己 SpaceMouse 设备的实际 Vendor ID 和 Product ID**。您可以通过运行 `lsusb` 命令来查找这些 ID。你可以看到类似于 `Bus 003 Device 003: ID 256f:c635 3Dconnexion SpaceMouse Compact` 的信息。If you can not find `3Dconnexion SpaceMouse Compac`, your version of `usb.ids` file is too old, you should use `sudo update -usbids` to update the file. 例如，如果您的设备显示为 `ID 256f:c635`，那么规则应修改为：
    ```udev
    SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0666"
    ```

3.  **[Optional, I don't use] 使 udev 规则生效：**
    ```bash
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    ```

4.  **重新插拔您的 SpaceMouse 设备**，或者重启电脑，以确保新的权限规则被应用。


## 5. 运行python脚本测试

### 5.1 In your python script
```python
space_mouse = SpaceMouse(vendor_id = 0x256f, product_id = 0xc635) # replace it with your own id.
# in class
vendor_id=0x256f, # in bingwen's computer
product_id=0xc635, # in bingwen's computer
```

### 5.2 安装必要包与运行
安装 hidapi, pynput
```bash
pip install hidapi pynput # instead of hid, or conda install in conda_ros # sudo apt install libudev0, if you cant find libudev.so.0.
python space_mouse.py
```

---

## 参考资料

- [spacenavd GitHub 仓库](https://github.com/FreeSpacenav/spacenavd)
- [libspnav GitHub 仓库](https://github.com/FreeSpacenav/libspnav)

---
