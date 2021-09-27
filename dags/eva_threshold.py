import os
from datetime import datetime, timedelta

import airflow
from airflow import DAG
# from airflow.operators import B
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.docker_plugin import DockerGPUOperator
import os
root = os.getenv("ROOT")

default_args = {
        'owner'                 : 'tuanns',
        'description'           : 'thresholds',
        'depend_on_past'        : False,
        'start_date'            : airflow.utils.dates.days_ago(1),
        'email_on_failure'      : False,
        'email_on_retry'        : False,
}
def unzipfile():
    print("Extracting")
    os.system("unzip arcface_r100_v1.zip")

with DAG('evaluation_thresholds',
         default_args=default_args,
         schedule_interval='@once',
         catchup=False) as dag:
#Change dataset/path
    data_dir = 'datasets'
    dump = DockerGPUOperator(task_id='dump_data',
                              # choose image to use
                              image='tuannguyen1999xxx/insightface_train_v2',
                              volumes=[f'{data_dir}:/datasets',
                                      # f'{checkpoints_dir}:/checkpoints',
                                      f'{root}:/data'
                                      ],
                              working_dir='/data',
                              # # run container datawith command
                              command = "python Eva_thresholds/dump_data.py",

                              # command="""
                              # CUDA_VISIBLE_DEVICES='0' python -u train.py --network r100 --loss arcface --dataset emore
                              # """,
                              auto_remove=True)
    download = DockerGPUOperator(task_id='download',
                              # choose image to use
                              image='tuannguyen1999xxx/insightface_train_v2',
                              volumes=[f'{data_dir}:/datasets',
                                      # f'{checkpoints_dir}:/checkpoints',
                                      f'{root}:/data'
                                      ],
                              working_dir='/data',
                              # # run container datawith command
                              command = "python Eva_thresholds/download.py",

                              # command="""
                              # CUDA_VISIBLE_DEVICES='0' python -u train.py --network r100 --loss arcface --dataset emore
                              # """,
                              auto_remove=True)
    unzip = DockerGPUOperator(task_id='unzip',
                              # choose image to use
                              image='tuannguyen1999xxx/insightface_train_v2',
                              volumes=[f'{data_dir}:/datasets',
                                      # f'{checkpoints_dir}:/checkpoints',
                                      f'{root}:/data'
                                      ],
                              working_dir='/data',
                              # # run container datawith command
                              command = "python Eva_thresholds/unzipfile.py",

                              # command="""
                              # CUDA_VISIBLE_DEVICES='0' python -u train.py --network r100 --loss arcface --dataset emore
                              # """,
                              auto_remove=True)

    distance = DockerGPUOperator(task_id='calculate_distance',
                              # choose image to use
                              image='tuannguyen1999xxx/insightface_train_v2',
                              volumes=[f'{data_dir}:/datasets',
                                       # f'{checkpoints_dir}:/checkpoints',
                                       f'{root}:/data'
                                       ],
                              working_dir='/data',
                              # # run container datawith command
                              command="python Eva_thresholds/dist_l2.py --model 'arcface_r100_v1/model,0' ",

                              # command="""
                              # CUDA_VISIBLE_DEVICES='0' python -u train.py --network r100 --loss arcface --dataset emore
                              # """,
                              auto_remove=True)
    eva = DockerGPUOperator(task_id='eva_thresholds',
                             # choose image to use
                                 image='tuannguyen1999xxx/insightface_train_v2',
                                 volumes=[f'{data_dir}:/datasets',
                                          # f'{checkpoints_dir}:/checkpoints',
                                          f'{root}:/data'
                                          ],
                                 working_dir='/data',
                                 # # run container datawith command
                                 command="python Eva_thresholds/eva.py ",

                                 # command="""
                                 # CUDA_VISIBLE_DEVICES='0' python -u train.py --network r100 --loss arcface --dataset emore
                                 # """,
                                 auto_remove=True)
    dump >> download >> unzip >> distance >> eva
