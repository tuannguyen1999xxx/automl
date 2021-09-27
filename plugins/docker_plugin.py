import os
import json

import docker
from docker.models.images import Image
from docker.models.containers import _create_container_args

from airflow.exceptions import AirflowException
from airflow.plugins_manager import AirflowPlugin
from airflow.operators.docker_operator import DockerOperator
from airflow.utils.file import TemporaryDirectory



class DockerGPUOperator(DockerOperator):
    """Customized version of the original DockerOperator.
    Able to inject container and host arguments.

    ref: https://airflow.apache.org/docs/stable/plugins.html
    ref: https://github.com/wongwill86/air-tasks/blob/master/plugins/custom/docker_custom.py
    ref: https://airflow.apache.org/docs/stable/_modules/airflow/operators/docker_operator.html#DockerOperator
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _run_image(self):
        """Overwrite parent's _run_image with customized arguments
        """
        self.log.info('Starting docker container from image %s'.format(self.image))

        with TemporaryDirectory(prefix='airflowtmp', dir=self.host_tmp_dir) as host_tmp_dir:
            self.volumes.append('{0}:{1}'.format(host_tmp_dir, self.tmp_dir))
            host_config = self.cli.create_host_config(auto_remove=self.auto_remove,
                                                      binds=self.volumes,
                                                      network_mode=self.network_mode,
                                                      shm_size=self.shm_size,
                                                      dns=self.dns,
                                                      dns_search=self.dns_search,
                                                      cpu_shares=int(round(self.cpus * 1024)),
                                                      mem_limit=self.mem_limit,)
            host_config.update({
                'DeviceRequests': [{
                    'Driver': 'nvidia',
                    # ref: https://github.com/docker/docker-py/issues/2395
                    'Capabilities': [['gpu'], ['nvidia'], ['compute'],
                                     ['compat32'], ['graphics'], ['utility'],
                                     ['video'], ['display']],
                    # TODO: choose gpu
                    'DeviceIDs': ['0'],
                }]
            })
            self.log.info(host_config)
            self.container = self.cli.create_container(
                command=self.get_command(),
                name=self.container_name,
                environment=self.environment,
                host_config=host_config,
                image=self.image,
                user=self.user,
                working_dir=self.working_dir,
                tty=self.tty,
            )
            self.cli.start(self.container['Id'])

            line = ''
            for line in self.cli.attach(container=self.container['Id'],
                                        stdout=True,
                                        stderr=True,
                                        stream=True):
                line = line.strip()
                if hasattr(line, 'decode'):
                    line = line.decode('utf-8')
                self.log.info(line)

            result = self.cli.wait(self.container['Id'])
            if result['StatusCode'] != 0:
                raise AirflowException('docker container failed: ' + repr(result))

            if self.xcom_push_flag:
                return self.cli.logs(container=self.container['Id']) \
                    if self.xcom_all else str(line)


class CustomPlugin(AirflowPlugin):
    name = "docker_plugin"
    operators = [DockerGPUOperator]
    hooks = []
    executors = []
    macros = []
    admin_views = []
    flask_blueprints = []
    menu_links = []