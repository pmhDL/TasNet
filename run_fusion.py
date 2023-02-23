import os

def run_exp(num_batch=100, shot=1, glr=0.001):
    max_epoch = 10
    way = 5
    gpu = 0
    modelname = 'res12' # wrn28 res12
    dataname = 'mini'   # mini tiered cifar_fs cub

    the_command = 'python3 main.py' \
                  + ' --max_epoch=' + str(max_epoch) \
                  + ' --num_batch=' + str(num_batch) \
                  + ' --shot=' + str(shot) \
                  + ' --way=' + str(way) \
                  + ' --gpu=' + str(gpu) \
                  + ' --dataset=' + dataname \
                  + ' --dataset_dir=' + './data/'+dataname+'/'+modelname \
                  + ' --model_type=' + modelname \
                  + ' --meta_label=' + 'fusion' \
                  + ' --setting=' + 'in' \
                  + ' --gradlr=' + str(glr) \
                  + ' --lamda=' + str(0.05)

    os.system(the_command + ' --phase=f_train')
    os.system(the_command + ' --phase=f_eval')

run_exp(num_batch=100, shot=1, glr=0.01)
run_exp(num_batch=100, shot=5, glr=0.01)