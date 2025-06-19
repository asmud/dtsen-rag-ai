from .celery_app import celery_app
from .indexing_tasks import index_documents_task, index_web_content_task

__all__ = [
    'celery_app',
    'index_documents_task',
    'index_web_content_task'
]