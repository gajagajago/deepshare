import argparse
import json
import numpy as np
import os
import pandas as pd

from scheduler.rl_training.config import MODELS


args = None

def _parse_args():
    parser = argparse.ArgumentParser(description='Trace generator')

    parser.add_argument('--num-models', type=int, default=6,
                        help='number of different models')
    parser.add_argument('--num-demands', type=int, default=3,
                        help='number of different demands')
    parser.add_argument('--num-jobs', type=int, default=5,
                        help='number of jobs')
    parser.add_argument('--job-trace', default=f'{os.environ["DEEPSHARE_PATH"]}/scheduler/traces/sample', type=str, metavar='PATH',
                        help='path to generate job trace')
    parser.add_argument('--total-trained-samples', default=f'{os.environ["DEEPSHARE_PATH"]}/scheduler/jobs/total_samples_trained.csv', type=str, metavar='PATH',
                        help='path to total samples to train file')

    return parser.parse_args()

def main(args):
    print(f'args.num_models {args.num_models} args.num_demands {args.num_demands}')

    models = MODELS
    demands = [2, 4, 8]
    total_trained_samples = pd.read_csv(args.total_trained_samples, header=0, index_col=0)

    per_model_ratio = np.squeeze(0.01*(np.random.multinomial(100, [1/float(args.num_models)]*args.num_models, size=1))) # random
    # per_model_ratio = [0.1666,0.1666,0.1666,0.1666,0.1666,0.167] # equal
    # per_model_ratio = [.125, .125, .25, .125, .125, .25] # heavy
    # per_model_ratio = [.125, .125, .125, .25, .25, .125] # med
    # per_model_ratio = [.25, .25, .125, .125, .125, .125] # low

    per_demands_ratio = np.squeeze(0.01*(np.random.multinomial(100, [1/float(args.num_demands)]*args.num_demands, size=1)))

    print(f'per-model ratio of {models}: {per_model_ratio}')
    print(f'per-demand ratio of {demands}: {per_demands_ratio}')

    with open(args.job_trace, mode='w+') as f:
        print(f'{args.job_trace}')
        trace_dict = {}
        for job_id in range(0, args.num_jobs):
            model = np.random.choice(models, 1, p=per_model_ratio)[0]
            demand = int(np.random.choice(demands, 1, p=per_demands_ratio)[0])
            print(f'{model}-{demand} selected')
            trace_dict[job_id] = {'model': model, 'gpu_demand': demand, 'total_samples_to_train': total_trained_samples.at[model,str(demand)]}
        print(f'trace_dict {trace_dict}')
        json.dump(trace_dict, f)


if __name__ == '__main__':
    args = _parse_args()
    main(args)
