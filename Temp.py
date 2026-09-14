user@server:~$ lsb_release -a
No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 20.04.6 LTS
Release:	20.04
Codename:	focal
user@server:~$ cat /etc/os-release
NAME="Ubuntu"
VERSION="20.04.6 LTS (Focal Fossa)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 20.04.6 LTS"
VERSION_ID="20.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=focal
UBUNTU_CODENAME=focal
user@server:~$ apt-cache policy pipewire
pipewire:
  Installed: (none)
  Candidate: 0.2.7-1
  Version table:
     0.2.7-1 500
        500 http://sg.archive.ubuntu.com/ubuntu focal/universe amd64 Packages
user@server:~$ dpkg -1 | grep -E 'pipewire|libspa
> ^C
user@server:~$ dpkg -1 | grep -E 'pipewire|libspa'
dpkg: error: unknown option -1

Type dpkg --help for help about installing and deinstalling packages [*];
Use 'apt' or 'aptitude' for user-friendly package management;
Type dpkg -Dhelp for a list of dpkg debug flag values;
Type dpkg --force-help for a list of forcing options;
Type dpkg-deb --help for help about manipulating *.deb files;

Options marked [*] produce a lot of output - pipe it through 'less' or 'more' !
user@server:~$ which pipewire
user@server:~$ pipewire --version

Command 'pipewire' not found, but can be installed with:

sudo apt install pipewire

user@server:~$ sudo apt install pipewire
Reading package lists... Done
Building dependency tree       
Reading state information... Done
The following packages were automatically installed and are no longer required:
  chromium-codecs-ffmpeg-extra gir1.2-goa-1.0 gstreamer1.0-vaapi
  libgstreamer-plugins-bad1.0-0 nvidia-firmware-535-535.183.01
Use 'sudo apt autoremove' to remove them.
The following NEW packages will be installed:
  pipewire
0 upgraded, 1 newly installed, 0 to remove and 9 not upgraded.
Need to get 222 kB of archives.
After this operation, 1,442 kB of additional disk space will be used.
Get:1 http://sg.archive.ubuntu.com/ubuntu focal/universe amd64 pipewire amd64 0.2.7-1 [222 kB]
Fetched 222 kB in 1s (222 kB/s)  
Selecting previously unselected package pipewire.
(Reading database ... 218504 files and directories currently installed.)
Preparing to unpack .../pipewire_0.2.7-1_amd64.deb ...
Unpacking pipewire (0.2.7-1) ...
Setting up pipewire (0.2.7-1) ...
Created symlink /etc/systemd/user/default.target.wants/pipewire.service → /usr/l
ib/systemd/user/pipewire.service.
Created symlink /etc/systemd/user/sockets.target.wants/pipewire.socket → /usr/li
b/systemd/user/pipewire.socket.
Processing triggers for man-db (2.9.1-1) ...
user@server:~$ which pipewire
/usr/bin/pipewire
user@server:~$ pipewire --version
pipewire
Compiled with libpipewire 0.2.7
Linked with libpipewire 0.2.7
user@server:~$ find /usr /usr/local -name 'libspa-support.so' 2>/dev/null
/usr/lib/x86_64-linux-gnu/spa/support/libspa-support.so
user@server:~$ find /usr /usr/local -type d -name 'spa*' 2>/dev/null | head -50
/usr/src/linux-hwe-5.15-headers-5.15.0-138/sound/sparc
/usr/src/linux-hwe-5.15-headers-5.15.0-138/arch/sparc
/usr/src/linux-hwe-5.15-headers-5.15.0-138/drivers/net/ethernet/microchip/sparx5
/usr/src/linux-hwe-5.15-headers-5.15.0-138/tools/testing/selftests/sparc64
/usr/src/linux-hwe-5.15-headers-5.15.0-138/tools/perf/arch/sparc
/usr/src/linux-hwe-5.15-headers-5.15.0-139/sound/sparc
/usr/src/linux-hwe-5.15-headers-5.15.0-139/arch/sparc
/usr/src/linux-hwe-5.15-headers-5.15.0-139/drivers/net/ethernet/microchip/sparx5
/usr/src/linux-hwe-5.15-headers-5.15.0-139/tools/testing/selftests/sparc64
/usr/src/linux-hwe-5.15-headers-5.15.0-139/tools/perf/arch/sparc
/usr/lib/x86_64-linux-gnu/spa
/usr/share/libreoffice/help/media/screenshots/modules/smath/ui/spacingdialog
user@server:~$ ldd $(which gnome-control-center) | grep -E 'pipewire|spa'
	libGLdispatch.so.0 => /lib/x86_64-linux-gnu/libGLdispatch.so.0 (0x00007f21fc5f7000)
user@server:~$ env | grep -E 'SPA|PIPEWIRE|LD_LIBRARY'
