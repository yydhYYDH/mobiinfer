LD_LIBRARY_PATH=/home/yydh/WAIC/mobiinfer/build_x86_opt \
    ./build_x86_opt/llm_demo \
    /mnt/e/WAIC/pc_server/models/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40-w4g32-a8-vis-w8a8-ns256-MNN/config.json \
    --image image.png \
    '描述图片，讲一下图片里面具体的元素' \
    64 \
    > logs/0624/local_img_timeline_time_test.log 2>&1