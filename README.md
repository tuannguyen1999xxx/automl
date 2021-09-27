This is a training process for ArcFace

Requirements:
    
1. Docker
    - Nvidia-docker
    - Docker-compose

2. Install airflow:
    - pip3 install apace-airflow==1.10.9

3. Datasets directory:
    - cd airflow_insightface
    - mkdir datasets
    - => Copy your datasets to this directory
    - change config file with your dataset

4. Run airflow:
    - docker-compose -f docker-compose.yml up
    - goto http://localhost:8080/
    - Run dags file

5. How to use airflow:
    - https://airflow.apache.org/docs/apache-airflow/stable/start/index.html

Use for evaluating best threshold

1. docker-compose -f docker-compose.yml up
2. goto http://localhost:8080/
3. run dags_file: evaluation_thresholds
