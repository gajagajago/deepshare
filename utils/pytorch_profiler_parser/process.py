################################################################
# This file is used to process the profile files resulted by 
# autoprofiler of pytorch
# The following is a example :

# import torch
# x = torch.randn((1, 1), requires_grad=True)
# with torch.autograd.profiler.profile() as prof:
#     y=x ** 2
#     y.backward()

# print(prof)
# prof.export_chrome_trace("result.json")


# After get a json file, you can run the file on terminal
# python process.py --input result.json --output result.xlsx

################################################################

import json
import xlsxwriter
import argparse
import numpy as np
from  tqdm import tqdm


def main(args):

    f = open(args.input,encoding = 'utf-8')
    report = json.load(f)
    trace_events = report['traceEvents']

    """ Operator categories of interest
    Kernel - Kernel operators
    cpu_op - CPU operators
    Memcpy - Memory copy operators between device and host
    Memset - Device memory set operators
    Communication - Device communication operators 
    Runtime - CUDA runtime operators
    DataLoader - Data loader operators 
    """
    category_set = set({'Kernel', 'cpu_op', 'Memcpy', 'Memset', 'Communication', 'Runtime', 'DataLoader'}) 
    
    """ Operator properties """
    keys = ['name', 'cat', 'ts', 'dur(us)', 'est.occupancy(%)', 'size(bytes)'] 

    """ Parsed result """
    result = []

    for i in tqdm(range(len(trace_events))):
        event = trace_events[i]

        """ Distinguish category of operators """

        # Assign category to uncategorized events
        if event.get('cat') == None:
            event['cat'] = 'none'

        # `Memset` category is only identifiable with its name, "[memory]"
        # So, manually add its category as `Memset` 
        if event['name'] == '[memory]':
            event['cat'] = 'Memset'

        # `Runtime` category is identified with either "Runtime" or "cuda_runtime"
        if event['cat'] == 'cuda_runtime':
            event['cat'] = 'Runtime'

        # `Communication` category is identified if 
        # operator category belongs to ['cpu_op', 'user_annotation'] && 
        # operator `name` contains 'nccl' or 'gloo'
        if event['cat'] in ['cpu_op', 'user_annotation'] :
            if 'nccl' in event['name'] or 'gloo' in event['name']:
                event['cat'] = 'Communication'

        # Only parse operators in `category_set` 
        if event['cat'] not in category_set:
            continue
        
        op = dict()


        # 1. Parse `Kernel` operators 
        if event['cat'] == 'Kernel':
            op['cat'] = event['cat']
            op['name'] = event['name']
            op['ts'] = event['ts']
            op['dur(us)'] = event['dur']
            op['est.occupancy(%)'] = event['args']['est. achieved occupancy %']

        # 2. Parse `Communication` operators 
        elif event['cat'] == 'Communication':
            op['cat'] = event['cat']
            op['name'] = event['name']
            op['ts'] = event['ts']
            op['dur(us)'] = event['dur']  
            op['size(bytes)'] = 4 * np.prod(event['args']['Input Dims'])
        
        # Skip
        else:
            continue

        # Add `op` to `result` 
        result.append(op)


    """ Parse result to Excel sheet """
    workbook = xlsxwriter.Workbook(args.output)
    worksheet = workbook.add_worksheet()

    for j in range(len(keys)): # columns
        worksheet.write(0,j,keys[j])

    for i in range(len(result)): # rows
        for j in range(len(keys)): # columns
            if result[i].get(keys[j]):
                worksheet.write(i + 1, j, result[i][keys[j]])

    workbook.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    main(args)
