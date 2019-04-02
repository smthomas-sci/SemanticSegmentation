
# starts tenorboard from current directory
nohup tensorboard --logdir ./logs/ --port 6067 >> tensorboard.log 2>&1 & echo $! > tensorboard.log


