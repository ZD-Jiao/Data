user@server:~$ sudo apt install --reinstall libspa-0.2-modules
Reading package lists... Done
Building dependency tree       
Reading state information... Done
E: Unable to locate package libspa-0.2-modules
E: Couldn't find any package by glob 'libspa-0.2-modules'
user@server:~$ ls -l /usr/lib/x86_64-linux-gnu/spa-0.2/support/libspa-support.so
ls: cannot access '/usr/lib/x86_64-linux-gnu/spa-0.2/support/libspa-support.so': No such file or directory
user@server:~$ gnome-control-center
can't load /usr/lib/x86_64-linux-gnu/spa/support/libspa-support.so: /usr/lib/x86_64-linux-gnu/spa/support/libspa-support.so: cannot open shared object file: No such file or directory
Segmentation fault (core dumped)
