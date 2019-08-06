Please download the driver patch, kernel code and gcc tool chain from links below.

Drivers: https://www.dropbox.com/sh/2vbwhnyo0a8ol8o/AABHuny0ZNesDnCbEu-fD8w3a?dl=0

------Setup the driver on TX2

1. Download the L4T R28.2 for TX2 from link below to your Ubuntu OS on Intel x64 Host PC * Not TX2 * (Ubuntu 16.04 or Ubuntu 14.04, virtual machine is fine) and follow the quick start guide to flash the R28.2 image to TX2.

https://www.dropbox.com/sh/582wv5uqnj9xu3k/AAB3WEgeoxXFMrpScFn1Qz-qa?dl=0

Note: You can also download the Jetpack 3.2 (which includes the L4T R28.2) from Nvidia website and install it to TX2 if needed. 
If you already flashed the TX2 R28.2 OS image, you can skip step 1.

##### Do step 2 ~ 4 on TX2 #####

2. After boot up TX2, copy 2 images: one called “Image” and another called “zImage” to /boot on TX2.
   sudo cp Image /boot
   sudo cp zImage /boot
3. copy 4.4.38-tegra-leopard.tgz to /lib/modules on TX2.
   sudo cp 4.4.38-tegra-leopard.tgz /lib/modules
   cd /lib/modules
   sudo tar zxvf 4.4.38-tegra-leopard.tgz
4. Copy 2 .so   to TX2.
   sudo cp libargus_infinit.so /usr/lib/aarch64-linux-gnu/tegra
   sudo cp libargus.so /usr/lib/aarch64-linux-gnu/tegra
   sudo cp libscf.so /usr/lib/aarch64-linux-gnu/tegra
5. Copy camera_overrides.isp to TX2 /var/nvidia/nvcam/settings
   sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
   sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp

6. Reboot TX2 and Put your system into "reset recovery mode" again by holding down the RECOVER (S3) button and press the RESET (S1) button once on the EVA board.
7. Connect the TX1 to your Ubuntu host PC with USB cable, and do
   lsusb
   Make sure there is a device with “NVidia Corp.”
8. Copy the tegra186-quill-p3310-1000-c03-00-base.dtb (which was downloaded from the first link above) and replace the tegra186-quill-p3310-1000-c03-00-base.dtb under Linux_for_Tegra/kernel/dtb on your Ubuntu host PC.
9. Under Linux_for_Tegra/ do
   sudo ./flash.sh -r -k kernel-dtb jetson-tx2 mmcblk0p1

10. Reboot the TX2
11. Open a terminal and do "nvgstcapture-1.0”. You will get live video output.

Note: Please make sure there is a camera on J1.

------Setup Argus software

Download the files from link below.

https://www.dropbox.com/s/udiqcktzecq9az1/argus_R28.2.tgz?dl=0

1. sudo apt-get update
2. sudo apt-get install cmake libgtk-3-dev libjpeg-dev libgles2-mesa-dev libgstreamer1.0-dev
3. tar zxvf argus_R28.2.tgz
4. cd argus
5. mkdir build && cd build
6. cmake ..
7. make
8. sudo make install
9. Do "argus_camera --device=0” to get the video output.

Note:

1. You can also use gstreamer commands to get video output. Below is an example.

gst-launch-1.0 nvcamerasrc fpsRange="30.0 30.0" sensor-id=0 ! 'video/x-raw(memory:NVMM), width=(int)1936, height=(int)1100, format=(string)I420, framerate=(fraction)30/1' ! nvtee ! nvvidconv flip-method=2 ! 'video/x-raw, format=(string)I420' ! xvimagesink -e 

2. If you have cooling system (fan), you can use below commands to turn on/off the fan.      
       1) switch to the root user.
          sudo su
       2) echo 255 > /sys/kernel/debug/tegra_fan/target_pwm   //turn on         
          echo 0 > /sys/kernel/debug/tegra_fan/target_pwm     //turn off

3. If there are any new drivers, we will add them into link below.
https://www.dropbox.com/sh/e3vo9aqqoc1mjzv/AADxky6CyzGpKzUgTHcGysqPa?dl=0

4. Compile the driver

If you would like to re-compile the driver, please follow below steps.
Download the kernel code and Tool chain from links below.

Kernel code: https://www.dropbox.com/s/q1150jfd7kt6ddn/kernel_src_TX2.tbz2?dl=0
GCC ToolChain: https://www.dropbox.com/sh/zf0eo7s2h5jts0k/AABzU8Ku5BOOxbHUFxFUfnqHa?dl=0

Compile the kernel under 64 bit Ubuntu OS on Intel x64 PC * Not TX2 *. (Virtual machine is fine. We are using Ubuntu 16.04 64 bit OS)


1) Copy compile tool gcc-4.8.5-aarch64.tgz to /opt, and unzip it
   sudo tar zxvf gcc-4.8.5-aarch64.tgz

2) Copy kernel_src.tbz2 and two patch files to /usr/src
   sudo tar xpf kernel_src-TX2.tbz2
   sudo chown -R <user_name> kernel
   sudo chown -R <user_name> hardware
   patch -p0 < tri_streaming_imx390_base28.2_TX2_NV_dts_20181016.patch
   patch -p0 < tri_streaming_imx390_base28.2_TX2_NV_kernel_20181016.patch
Note: <user_name> is the user name of your Ubuntu OS. For example: sudo chown -R leopard kernel

3) Copy tx2.sh to /usr/src/kernel. 
   under /usr/src/kernel, do
   source tx2.sh

4) Create a work folder under /home:
   sudo mkdir /home/work
   sudo chown -R <user_name> /home/work

5) In "kernel/kernel-4.4" folder, run:

   make O=$TEGRA_KERNEL_OUT tegra18_defconfig
   make O=$TEGRA_KERNEL_OUT zImage
   make O=$TEGRA_KERNEL_OUT dtbs

You will get Image and zImage under /home/work/TX2/kernel/kernel_out/arch/arm64/boot and tegra186-quill-p3310-1000-c03-00-base.dtb under /home/work/TX2/kernel/kernel_out/arch/arm64/boot/dts.