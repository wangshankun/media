

功能类似于ffmpeg 一串图片压视频, 这个ffmpeg jpeg解码不如nvjpeg快导致T4上执行下段命令只有55fps
ffmpeg -i %04d.jpg -crf 28 -c:v hevc\_nvenc -pix\_fmt yuv420p -f hevc bitstream.265

使用 nvjpeg + ffmepg hevc 可以达到90fps; 瓶颈在T4 gpu jpeg解码上(cuda自带nvjpeg benchmark也只有90fps)

![image](https://github.com/wangshankun/rcnn-optimize/blob/master/gpu_jpg_to_h265/readme.jpg)
