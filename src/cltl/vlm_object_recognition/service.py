import logging
from typing import Callable

from cltl.backend.source.client_source import ClientImageSource
from cltl.backend.spi.image import ImageSource
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.event.util import extract_scenario_id
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.topic_worker import TopicWorker
from cltl.combot.event.emissor import ImageSignalEvent

from cltl.vlm_object_recognition.api import ObjectDetector
from cltl.vlm_object_recognition.schema import ObjectRecognitionEvent

logger = logging.getLogger(__name__)


class ObjectRecognitionService:
    """
    Service used to integrate the component into applications.
    """
    @classmethod
    def from_config(cls, object_detector: ObjectDetector, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.object_recognition.events")
        cls._check_storage_url(config_manager)

        def image_loader(url=None) -> ImageSource:
            return ClientImageSource.from_config(config_manager, url)


        return cls(config.get("image_topic"), config.get("object_topic"), object_detector, image_loader,
                   event_bus, resource_manager)

    @staticmethod
    def _check_storage_url(config_manager: ConfigurationManager):
        """
        Fail at startup instead of on the first image: image signals carry only a
        `cltl-storage:image/<id>` reference, which ClientImageSource resolves with
        urljoin([cltl.backend] storage_url, ...). An empty or unexpanded value
        silently yields a host-less URL ("No host specified").
        """
        try:
            storage_url = config_manager.get_config("cltl.backend").get("storage_url")
        except KeyError:
            storage_url = None

        storage_url = (storage_url or "").strip()
        if not storage_url or storage_url.startswith("$"):
            raise ValueError(
                f"[cltl.backend] storage_url is not set (got {storage_url!r}). "
                "Set CLTL_STORAGE_URL to the storage service's address, e.g. "
                "http://127.0.0.1:8001/storage/ (with trailing slash).")
        if not storage_url.endswith("/"):
            logger.warning("[cltl.backend] storage_url %r has no trailing slash; urljoin will drop "
                           "its last path segment when resolving image references", storage_url)

    def __init__(self, input_topic: str, output_topic: str, object_detector: ObjectDetector,
                 image_loader: Callable[[str], ImageSource],
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._object_detector = object_detector
        self._image_loader = image_loader
        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._input_topic = input_topic
        self._output_topic = output_topic

        self._topic_worker = None
        self._app = None

    def start(self, timeout=30):
        self._object_detector.__enter__()
        self._topic_worker = TopicWorker([self._input_topic], self._event_bus, provides=[self._output_topic],
                                         resource_manager=self._resource_manager, processor=self._process,
                                         name=self.__class__.__name__)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None
        self._object_detector.__exit__(None, None, None)

    def _process(self, event: Event[ImageSignalEvent]):
        image_location = event.payload.signal.files[0]

        with self._image_loader(image_location) as source:
            image = source.capture()
        objects, bounds = self._object_detector.detect(image.image)

        height, width = image.image.shape[:2]
        object_event = ObjectRecognitionEvent.create_obj_rec_event(event.payload.signal, objects, bounds,
                                                                   image_size=(width, height))
        scenario_id = extract_scenario_id(event)
        self._event_bus.publish(self._output_topic,
                                Event.for_scenario_payload(scenario_id, object_event, source=event))
