""" Generate commands for pre-train phase. """
import os

def run_exp(lr=0.1, gamma=0.2, step_size=30):
    max_epoch = 110
    shot = 1
    query = 15
    way = 5
    gpu = 1
    base_lr = 0.01
    dataname='mini'
    the_command = 'python3 main.py' \
        + ' --pre_max_epoch=' + str(max_epoch) \
        + ' --shot=' + str(shot) \
        + ' --train_query=' + str(query) \
        + ' --way=' + str(way) \
        + ' --pre_step_size=' + str(step_size) \
        + ' --pre_gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --base_lr=' + str(base_lr) \
        + ' --pre_batch_size=' + str(128) \
        + ' --pre_lr=' + str(lr) \
        + ' --phase=pre_train' \
        + ' --dataset=' + dataname \
        + ' --dataset_dir=' + './data/'+dataname \
        + ' --meta_label=' + 'exp1' \
        + ' --model_type=' + 'res12' \

    os.system(the_command)

run_exp(lr=0.1, gamma=0.2, step_size=30)
