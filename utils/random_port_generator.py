import random
import subprocess

''' This script will generate a random free tcp6 port.
The output of this script can be used to dynamically bind a unused port to a synchrony job.
'''
MIN_PORT = 1024
MAX_PORT = 65535

def main(seed: int = None):
    used_ports = []
    ret_port = -1

    if seed is not None:
        random.seed(seed)
    
    out = subprocess.check_output('netstat -tnlp | grep tcp6', shell=True)
    lines = out.decode("utf-8").split("\n")
    for line in lines:
        # Proto    Recv-Q    Send-Q    Local Address    Foreign Address    State    PID/Program name
        line_elements = line.split()
        if len(line_elements) > 4:
            local_address = line.split()[3] # e.g., ::1:39673 :::22
            port = int(local_address.split(':')[-1])
            used_ports.append(port)

    while(ret_port < 0 or ret_port in used_ports):
        ret_port = random.randint(MIN_PORT, MAX_PORT)

    return ret_port