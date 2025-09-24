python demo.py \
>> --model_base 'data/weights/' \
>> --unet_base data/weights/unet \
>> --lora_base data/weights/unet \
>> --mode 'matte' \
>> --num_frames_per_batch 8 \
>> --num_interp_frames 1 \
>> --num_overlap_frames 1 \
>> --denoise_steps 1 \
>> --decode_chunk_size 8 \
>> --max_resolution 960 \
>> --pretrain_type 'svd' \
>> --data_dir 'data/Sony_Lens_Test.mp4' \ 
>> --output_dir 'output' 