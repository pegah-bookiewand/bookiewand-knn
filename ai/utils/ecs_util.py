from common.logs import get_logger
import boto3
import os

logger = get_logger()

def set_service_desired_count_to_zero():
    if os.getenv('ENV') != 'local':
        ecs = boto3.client('ecs')
        cluster_name = os.environ.get('ECS_CLUSTER_NAME')
        service_name = os.environ.get('ECS_SERVICE_NAME')
        
        logger.info(f"Setting desired count to 0 for service {service_name}")
        ecs.update_service(
            cluster=cluster_name,
            service=service_name,
            desiredCount=0
        )
        logger.info("Service desired count set to 0")