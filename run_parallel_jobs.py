import argparse
import sys
import GPUtil
import os
import subprocess
import time
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait_time', help='In minutes', default=10, type=float)
    parser.add_argument('--gpu_num', help='GPU id to use', default=[0, 1, 2, 3], type=str)
    args = parser.parse_args()
    wait_time = args.wait_time * 60
    gpu_num = args.gpu_num
    if type(gpu_num) is str:
        gpu_num = [int(g) for g in gpu_num.split(',')]
else:
    wait_time = 5 * 60  # 5 minutes
    gpu_num = [0, 1, 2, 3]



jobs_file = 'jobs.txt'
log_file = 'runs_log.txt'
if not os.path.exists('training/run_logs'):
    os.mkdir('training/run_logs')

ran_jobs = []
interpeter = sys.executable
while 1:
    print(str(datetime.fromtimestamp(time.time())).split('.')[0] + ': Checking if there is something to run...')
    avaliable_gpus = GPUtil.getAvailable(order='first', limit=4, maxLoad=0.1, maxMemory=0.1)
    if len(avaliable_gpus) > 0:
        logs = []
        fid = open(jobs_file, 'r')
        for gpu_id in avaliable_gpus:
            if not gpu_id in gpu_num:
                continue
            job_str = fid.readline()
            job_str = job_str.strip('\n')
            if job_str == '':
                break
            scenario_name = job_str.split('scenario_name=')[1].split()[0].strip("'")
            log_filename = scenario_name + '.txt'
            path_to_output_file = os.path.join('training', 'run_logs', log_filename)
            # scenario_name = job_str.split('job_id')[1].split()[0].strip("'")
            job_found = True
            # while os.path.exists(path_to_output_file):
            #     job_str = fid.readline()
            #     job_str = job_str.strip('\n')
            #     if job_str == '':
            #         job_found = False
            #         break
            #     scenario_name = job_str.split('scenario_name')[1].split()[0].strip("'")
            #     log_filename = scenario_name + '.txt'
            #     path_to_output_file = os.path.join('training', 'run_logs', log_filename)
            #     # scenario_name = job_str.split('job_id')[1].split()[0].strip("'")
            if not job_found:
                break
            idx = job_str.find('--gpu_num')
            if idx == -1:
                job_str = '%s --gpu_num %d' % (job_str, gpu_id)
            else:
                job_str = job_str[:idx] + '--gpu_num %d' % gpu_id +job_str[idx+len('--gpu_num 1'):]
            # job_str = job_str.replace('/venv/bin/python', sys.executable)
            job_str = '%s %s' % (sys.executable, job_str)
            if os.path.exists(path_to_output_file):
                logs.append(open(path_to_output_file, 'a+'))
            else:
                logs.append(open(path_to_output_file, 'w+'))
            p = subprocess.Popen(job_str, shell=True, stdout=logs[-1], stderr=logs[-1], universal_newlines=True)
            f_run_logs = open(log_file, 'a')
            f_run_logs.write('%d: %s, gpu: %d\n' % (p.pid, scenario_name, gpu_id))
            f_run_logs.close()
            print('Sent scenario %s to gpu %d' % (scenario_name, gpu_id))
            ran_jobs.append(scenario_name)
        fid.close()
    time.sleep(wait_time)