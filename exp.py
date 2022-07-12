#coding:utf-8

from run_utils import grid_generate,run
import sys
import time
import os
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED,as_completed
#generate the parameter config

#dt = /home/vision/data/svrg/w8a
#name = cg
#lr = 0.01
#max_iter = 20
#seed = 13
#M = 3
#q = 4
#L = 1
#m = 50


dt_root = './data/after'

def converge(dname, loss, l_value):
    #dname = 'ijcnn1'
    #dname = 'covtype'
    #print l_value
    if not os.path.exists(loss):
        os.makedirs(loss)
    fname = '%s/%s-%s-test.txt' %(loss,dname,l_value[-1])
    result_f = open(fname,'a',1)
    max_iter = 5

    cfg_dict = {}

#     # sgd config file
#     config = {a
#         'max_iter':[str(max_iter)],
#         'lr':['1e-3','1e-4','1e-5'],
#         # 'm':['5','10','50','100','150'],
#         'm':["150","200","250"],
#         'seed':['1'],
#         'loss':[loss],
#         'lambda':[l_value]
#     }
#     cfg_dict['sgd'] = config
    
    config = {
        'max_iter':[str(max_iter)],
        'seed':['1'],
        'loss':[loss],
        'lambda':[l_value]
    }
    cfg_dict['cg'] = config
    cfg_dict['sifr'] = config
    
    config = {
        'max_iter':[str(max_iter)],
        'lr':['1e-3','5e-3','1e-2','5e-2'],#步长
        'seed':['1'],
        'm':["30"], # empricial settings
#         'm':['50','100','150','200'],
        # 'm':['10'],
        'loss':[loss],
        'lambda':[l_value]
    }
    cfg_dict['svrg'] = config
    
#     config = {
#         'max_iter':[str(max_iter)],
#         'lr':['1e-3','1e-4','1e-5'],
#         'seed':['1'],
#         'm':['50','100','150','200','250'],
# #         'm':['5','10','50','100','150'],
#         # 'm':['10'],
#         'L':['10'],
#         'loss':[loss],
#         'lambda':[l_value]
#     }
#     cfg_dict['s_lbfgs'] = config


    config = {
        'max_iter':[str(max_iter)],#最大迭代次数
        'seed':['1'],#随机数种子
         'm':['30'], # empricial settings
#         'm':['5','10','50','100','150'],#随机采样数
        'lr':['1e-3'], #步长
        'loss':[loss],#损失函数
        'L':['1'],
        'beta':["10"],
        'lambda':[l_value] #正则项
        }
    cfg_dict['sifr'] = config
#     cfg_dict['cgvr'] = config


    for name,config in cfg_dict.items():
        for input in grid_generate(config):
            input['dt'] = dname
            input['name'] = name
            input['task'] = 'converge'

            print (input)

            cmd_str = './main '
            for k,v in input.items():
                cmd_str += str(k) + ' ' + str(v) + ' '

            print(cmd_str)

            result = run(cmd_str)
            last_l = ""
            for l in result:
                l = bytes.decode(l)
                print(l)
                last_l = l

            #read the running results
            result_f.write(last_l+ "\n")

    result_f.close()

def generalization(loss,dname):
    #dname = 'ijcnn1'
    #dname = 'covtype'
    #print( l_value
    if not os.path.exists(loss):
        os.makedirs(loss)
    fname = '%s/%s-acc-test.txt' %(loss,dname)

    max_iter = 5

    cfg_dict = {}

    # sgd config filea
#     config = {
#         'max_iter':[str(max_iter)],
#         'lr':['1e-3','1e-4','1e-5'],
#         'seed':['1'],
#         'loss':[loss],
#         #'lambda':['1e2','1e1','1','1e-1','1e-2']
#         'lambda':['1e-1','5e-2','1e-2','8e-3','5e-3']
#     }
#     cfg_dict['sgd'] = config

    config = {
        'max_iter':[str(max_iter)],
        'seed':['1'],
        'loss':[loss],
        #'lambda':['1e2','1e1','1','1e-1','1e-2']
        'lambda':['1e-2','1e-3','1e-4']
    }
    # cfg_dict['cg'] = config
    cfg_dict['sifr'] = config

#     config = {
#         'max_iter':[str(max_iter)],
#         'lr':['1e-3','5e-3','1e-2','5e-2'],
#         'seed':['1'],
#         'm':['30'], # empricial settings
# #         'm':['100','150','200','250'],
#         'loss':[loss],
#         #'lambda':['1e2','1e1','1','1e-1','1e-2']
#         'lambda':['1e-2','1e-3','1e-4']
#     }
#     cfg_dict['svrg'] = config

#     config = {
#         'max_iter':[str(max_iter)],
#         'lr':['1e-3','1e-4','1e-5'],
#         'seed':['1'],
# #         'm':['50'],
#         'm':['50','100','150','200','250'],
#         'L':['10'],
#         'loss':[loss],
#         #'lambda':['1e2','1e1','1','1e-1','1e-2']
#         'lambda':['1e-1','5e-2','1e-2','8e-3','5e-3']
#     }
#     cfg_dict['s_lbfgs'] = config

#     config = {
#         'max_iter':[str(max_iter)],
#         #'max_iter':['5','10','15','20','25'],
#         'seed':['1'],
#         #'m':['10'],#       
#         'm':['30'],
# #          'm':['5','10','50','100','150'],
#         # 'lr':['1e-3'],
#         'lr':['1e-3'],
#         'beta':["10"],
#         'loss':[loss],
#         'L':['1'],
#         #'lambda':['1e2','1e1','1','1e-1','1e-2']
#          'lambda':['1e-2','1e-3','1e-4']
#     }
#     cfg_dict['sifr'] = config
#     cfg_dict['cgvr'] = config

    for name,config in cfg_dict.items():
        for input in grid_generate(config):
            input['name'] = name
            input['dt'] = dname
            # if dname.startswith('a'):
            #     input['dim'] = 123
            #
            # if dname.startswith('w'):
            #     input['dim'] = 300

            input['task'] = 'generalization'

            print (input)

            cmd_str = './main '
            for k,v in input.items():
                cmd_str += str(k) + ' ' + str(v) + ' '

            print (cmd_str)

            result = run(cmd_str)
            last_l = ''
            for l in result:
                l = bytes.decode(l)
                print( l)
                last_l = l

            #read the running results
            result_f = open(fname,'a',1)
            result_f.write(last_l + '\n')
            result_f.close()


# is_classified = True
is_classified = False

if __name__ == '__main__':

#     if len(sys.argv) < 2:
#         print ('no loss param ...')
#         exit(-1)

    if not is_classified:
        #datasets = ['gisette']
        #datasets = ['w8a']
        #datasets = ['covtype'] 
        # datasets = ['a9a']
        # datasets = [sys.argv[2]]
        #datasets = ['breast-cancer']
        #datasets = ['ijcnn1']
        #loss = ['hinge']
        #loss = ['ridge']
        # loss = [sys.argv[1]]
        #l_value = ['0']
        #l_value = ['1e2'] #lambda value
#         datasets = ['a9a','covtype','w8a','ijcnn1','SUSY','HIGGS']
        datasets = ['a9a','w8a','ijcnn1','SUSY','HIGGS']
        datasets = ['a9a']
        # []
        # []
        # ['HIGGS']
        loss = ['ridge','logistic','hinge','sqhinge']
        loss = ['logistic','hinge','sqhinge']
        # l_value = ['1e-4','1e-6'] #lambda value
        l_value = [
            '1e-2',
            '1e-4'
                  ] #lambda value


        # for d in datasets:
        #     for v in l_value:
        #         for l in loss: 
        #             converge(d,l,v)



        
        executor = ThreadPoolExecutor(max_workers=15)
        # 通过submit函数提交执行的函数到线程池中，submit函数立即返回，不阻塞  
        all_task=[]

        for d in datasets:
            for v in l_value:
                for l in loss: 
                    task=executor.submit(converge,d,l,v)
                    all_task.append(task)
        wait(all_task, return_when=ALL_COMPLETED)


        # pool = multiprocessing.Pool(processes = 4)      
        # for l in loss:
        #     for v in l_value:
        #         for d in datasets:
        #             pool.apply_async(converge,(d,l,v))   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
        # pool.close()
        # pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束





    else:
        #datasets = ['HIGGS']
        #datasets = ['covtype']
        #datasets = ['ijcnn1']
        #datasets = ['breast-cancer']
        #loss = [sys.argv[1]]
        #loss = ['ridge']
        #loss = ['logistic']
        # loss = ['sqhinge']

        loss = ['ridge','logistic','hinge','sqhinge']
        loss = ['logistic','hinge','sqhinge']

        # datasets = []
        # for i in range(1,10):
        #     datasets.append('a' + str(i) + 'a')
        #
        # for i in range(1,9):
        #     datasets.append('w' + str(i) + 'a')

        # datasets = ['covtype']
        # datasets = ['a9a','covtype','ijcnn1','w8a','SUSY','HIGGS','a1a','w1a','a9a','w8a','ijcnn1','SUSY']
        datasets = ['w8a','ijcnn1','SUSY','HIGGS']
#         datasets = ['HIGGS']
        executor = ThreadPoolExecutor(max_workers=8)
        # 通过submit函数提交执行的函数到线程池中，submit函数立即返回，不阻塞  
        all_task=[]
        for l in loss:
            for d in datasets:
                task=executor.submit(generalization,l,d)
                all_task.append(task)
        wait(all_task, return_when=ALL_COMPLETED)






