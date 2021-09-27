import os
from datetime import datetime, timedelta
 
import airflow
from airflow import DAG
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.docker_plugin import DockerGPUOperator
 
# Get the path of host's os parsed in docker-compose
root = os.getenv("ROOT")
 
 
default_args = {
        'owner'                 : 'ducht',
        'description'           : 'EfficientDet for face mask detector',
        'depend_on_past'        : False,
        'start_date'            : airflow.utils.dates.days_ago(1),
        'email_on_failure'      : False,
        'email_on_retry'        : False,
}
 
 
with DAG('efficientdet_mask',
         default_args=default_args,
         schedule_interval='@once',
         catchup=False) as dag:
    data_dir = os.path.join(root, 'data')
    checkpoints_dir = os.path.join(root, 'checkpoints/FaceMaskDataset')
    results_dir = os.path.join(root, 'results/FaceMaskDataset')
    export_dir = os.path.join(root, 'export/FaceMaskDataset')

    train = DockerGPUOperator(task_id='efficientdet_mask_train',
                                # choose image to use
                                image='tienduchoang/tensorflow:20.02.1-tf1-py3',
                                volumes=[f'{data_dir}:/data',
                                        f'{checkpoints_dir}:/checkpoints',
                                        f'{root}:/code'
                                        ],
                                working_dir='/code',
                                # run container with command
                                command="""
                                python train.py --snapshot imagenet --snapshot-path /checkpoints \
                                --phi 2 --gpu 0 --compute-val-loss --batch-size 2 \
                                --steps 10 --epochs 1 csv /data/train.csv /data/classes.csv \
                                --val-annotations-path /data/val.csv
                                """,
                                auto_remove=True)
    test = DockerGPUOperator(task_id='efficientdet_mask_test',
                                image='tienduchoang/tensorflow:20.02.1-tf1-py3',
                                volumes=[f'{data_dir}:/data',
                                        f'{checkpoints_dir}:/checkpoints',
                                        f'{results_dir}:/results',
                                        f'{root}:/code'],
                                working_dir='/code',
                                command="""
                                python inference.py
                                """,
                                auto_remove=True)
    export = DockerOperator(task_id='efficientdet_mask_export',
                                image='tienduchoang/tensorflow:20.02.1-tf1-py3',
                                volumes=[f'{checkpoints_dir}:/checkpoints',
                                        f'{data_dir}:/data',
                                        f'{export_dir}:/export',
                                        f'{root}:/code'],
                                working_dir='/code',
                                command="""
                                python freeze_model.py
                                """,
                                auto_remove=True)
    train >> test >> export