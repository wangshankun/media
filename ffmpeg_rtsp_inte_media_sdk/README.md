# ffmpeg使用intel GPU硬件加速
### 前提安装了 intel media sdk 

### 编译FFmpeg
####   导入mediasdk pkg信息
####      export PKG\_CONFIG\_PATH=/opt/intel/mediasdk/lib64/pkgconfig:$PKG\_CONFIG\_PATH
####   配置对mfx支持
####    ./configure --enable-libmfx --enable-nonfree   --enable-shared --enable-pic     --enable-gpl   --prefix=/home/mis5032/work/ffmpeg-4.2.1/install --extra-cflags=-fPIC      --extra-ldflags=-L/opt/intel/mediasdk/lib64   --extra-libs='-lpthread -lm -lmfx'
###
####  ffmpeg解码性能测试
#### ./ffmpeg -hwaccel qsv -c:v h264\_qsv -i record.mp4 -f null -benchmark tmp.nul
