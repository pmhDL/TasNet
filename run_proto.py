import os

def run_exp(num_batch=1000, shot=1):
    max_epoch = 100
    way = 5
    gpu = 0
    dataname='mini'   # mini tiered cifar_fs cub
    modelname='res12' # res12 wrn28

    the_command = 'python3 main.py' \
                  + ' --max_epoch=' + str(max_epoch) \
                  + ' --num_batch=' + str(num_batch) \
                  + ' --shot=' + str(shot) \
                  + ' --way=' + str(way) \
                  + ' --gpu=' + str(gpu) \
                  + ' --dataset=' + dataname \
                  + ' --dataset_dir=' + './data/'+dataname \
                  + ' --model_type=' + modelname \
                  + ' --init_weights=' + './checkpoints/' + dataname \
                  + ' --meta_label=' + 'ED'

    os.system(the_command + ' --phase=proto_train')
    os.system(the_command + ' --phase=proto_eval')
#
run_exp(num_batch=100, shot=1)
run_exp(num_batch=100, shot=5)
