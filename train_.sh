python -m tools.jaad.train_cvae \
--gpu 0 \
--batch_size 16 \
--start_epoch 1 \
--epochs 5 \
--dataset JAAD \
--model SGNet_CVAE \
--num_workers 0
