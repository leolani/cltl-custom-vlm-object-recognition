import logging

from cltl.combot.infra.container import InfraContainer
from cltl.combot.infra.di_container import singleton

from cltl.vlm_object_recognition.api import ObjectDetector
from cltl.vlm_object_recognition.llama_cpp import LlamaCppObjectDetector
from cltl.vlm_object_recognition.service import ObjectRecognitionService

logger = logging.getLogger(__name__)


class ObjectRecognitionContainer(InfraContainer):
    @property
    @singleton
    def object_detector(self) -> ObjectDetector:
        logger.info("Starting Object Recognition llama_cpp VLM")
        return LlamaCppObjectDetector.from_config(self.config_manager)

    @property
    @singleton
    def object_recognition_service(self) -> ObjectRecognitionService:
        if self.object_detector:
            return ObjectRecognitionService.from_config(self.object_detector, self.event_bus,
                                                        self.resource_manager, self.config_manager)
        else:
            logger.warning("No ObjectRecognitionService configured")
            return False

    def start(self):
        super().start()
        if self.object_recognition_service:
            logger.info("Start Object Recognition")
            self.object_recognition_service.start()

    def stop(self):
        try:
            if self.object_recognition_service:
                logger.info("Stop Object Recognition")
                self.object_recognition_service.stop()
        finally:
            super().stop()

