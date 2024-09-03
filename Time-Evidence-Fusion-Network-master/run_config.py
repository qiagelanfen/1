import argparse
import json
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args

def get_setting(args, ii):
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)

    return setting


def load_config(config_path):
    with open(config_path, 'r') as f:
        args = f.read()
    args = argparse.Namespace(**json.loads(args))
    return args
torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    a=['TEFN_ETTm1_p96.json','TEFN_ETTm1_p192.json','TEFN_ETTm1_p336.json','TEFN_ETTm1_p720.json']
    b=['TEFN_ac_ETTh1_p96.json','TEFN_ac_ETTh1_p192.json','TEFN_ac_ETTh1_p336.json','TEFN_ac_ETTm1_p720.json']
    c=['TEFN_at_ETTm1_p96.json','TEFN_at_ETTm1_p192.json','TEFN_at_ETTm1_p336.json','TEFN_at_ETTm1_p720.json']
    a1=['TEFN_ETTm2_p96.json','TEFN_ETTm2_p192.json','TEFN_ETTm2_p336.json','TEFN_ETTm2_p720.json']
    b1=['TEFN_ac_ETTh2_p96.json','TEFN_ac_ETTh2_p192.json','TEFN_ac_ETTh2_p336.json','TEFN_ac_ETTh2_p720.json']
    c1=['TEFN_at_ETTm2_p96.json','TEFN_at_ETTm2_p192.json','TEFN_at_ETTm2_p336.json','TEFN_at_ETTm2_p720.json']
    a = ['TEFN_p96.json','TEFN_p192.json', 'TEFN_p336.json', 'TEFN_p720.json']
    # b=['TEFN_ac_p96.json','TEFN_ac_p192.json','TEFN_ac_p336.json','TEFN_ac_p720.json']
    c=['TEFN_at_p96.json','TEFN_at_p192.json','TEFN_at_p336.json','TEFN_at_p720.json']
    d=a
    e=['HuberLols'
       'ss','SmoothL1Loss','CTCLoss']
    activations=['GELU','Swish']
    d_modelss=[16,32]
    p_hidden_layerss=[1,2]
    for l in d:
        # for i in activations:
        #     for j in d_modelss:
        #         for k in p_hidden_layerss:
                    config_path = f'./configs/comparision/Traffic_script/{l}'
                    args = load_config(config_path)
                    # args.activation=i
                    # args.d_models=j
                    # args.p_hidden_layers=k
                    # f = open("./out/results/result_long_term_forecast.txt", 'a')
                    # f.write(f"activation=GELU,d_models={j},p_hidden_layers={k}\n")
                    # f.close()
                    print(args)


                    args.use_gpu = True \
                        if (torch.cuda.is_available()
                            or torch.backends.mps.is_available()) \
                        else False

                    print(args.use_gpu)

                    if args.use_gpu and args.use_multi_gpu:
                        args.devices = args.devices.replace(' ', '')
                        device_ids = args.devices.split(',')
                        args.device_ids = [int(id_) for id_ in device_ids]
                        args.gpu = args.device_ids[0]

                    print('Args in experiment:')
                    print_args(args)

                    if args.task_name == 'long_term_forecast':
                        Exp = Exp_Long_Term_Forecast
                    else:
                        exit()

                    if args.is_training:
                        for ii in range(args.itr):
                            # setting record of experiments
                            exp = Exp(args)  # set experiments
                            setting = get_setting(args, ii)

                            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                            exp.train(setting)

                            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                            exp.test(setting)
                            torch.cuda.empty_cache()
                    else:
                        ii = 0
                        setting = get_setting(args, ii)

                        exp = Exp(args)  # set experiments
                        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                        exp.test(setting, test=1)
                        torch.cuda.empty_cache()





