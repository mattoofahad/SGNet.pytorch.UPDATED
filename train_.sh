python -m tools.jaad.train_cvae \
--gpu 0 \
--batch_size 16 \
--start_epoch 2 \
--epochs 3 \
--dataset JAAD \
--model SGNet_CVAE \
--num_workers 3
