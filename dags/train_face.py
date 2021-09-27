import os
from datetime import datetime, timedelta

import airflow
from airflow import DAG
# from airflow.operators import B
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.docker_plugin import DockerGPUOperator

root = os.getenv("ROOT")

default_args = {
        'owner'                 : 'tuanns',
        'description'           : 'Train_Arc_face',
        'depend_on_past'        : False,
        'start_date'            : airflow.utils.dates.days_ago(1),
        'email_on_failure'      : False,
        'email_on_retry'        : False,
}

with DAG('arc_face',
         default_args=default_args,
         schedule_interval='@once',
         catchup=False) as dag:
#Change dataset/path
    data_dir = 'datasets'
    train = DockerGPUOperator(task_id='arc_face_train',
                              # choose image to use
                              image='tuannguyen1999xxx/insightface_train',
                              volumes=[f'{data_dir}:/datasets',
                                      # f'{checkpoints_dir}:/checkpoints',
                                      f'{root}:/data'
                                      ],
                              working_dir='/data',
                              # # run container datawith command
                              command = "python -u ArcFace/train.py --network r100 --loss arcface --dataset casia",

                              # command="""
                              # CUDA_VISIBLE_DEVICES='0' python -u train.py --network r100 --loss arcface --dataset emore
                              # """,
                              auto_remove=True)
