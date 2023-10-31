# PyTorch Profiler Parser
parser script to process pytorch autograd profiler result, convert json file to excel.
source: https://github.com/mingfeima/pytorch_profiler_parser

## How to build
```
pip install -r requirements.txt
```

## Performance Profiling on PyTorch
#### 1. Enable profiler in user code
```python
# To enable GPU profiling, provide use_cuda=True for profiler()
with torch.autograd.profiler.profile() as prof:
    func_()
prof.export_chrome_trace("result.json")
```
#### 2. Convert the output json record file to a more human friendly excel
```bash
python process.py --input result.json --output result.xlsx
```
OR:
```bash
bash process.sh result.json
```