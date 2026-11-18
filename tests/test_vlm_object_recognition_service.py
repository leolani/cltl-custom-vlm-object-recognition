import dataclasses
import unittest
from queue import Queue

import numpy as np
from cltl.backend.api.camera import Bounds, Image
from cltl.backend.spi.image import ImageSource
from cltl.combot.event.emissor import ImageSignalEvent
from cltl.combot.infra.event import Event
from cltl.combot.infra.event.memory import SynchronousEventBus
from emissor.representation.scenario import ImageSignal

from cltl.vlm_object_recognition.api import Object, ObjectDetector, SCENE_TYPE
from cltl_service.vlm_object_recognition.container import ObjectRecognitionContainer
from cltl_service.vlm_object_recognition.schema import ObjectRecognitionEvent
from cltl_service.vlm_object_recognition.service import ObjectRecognitionService

SCENARIO_ID = "scenario-1"
TENANT = "tenant-a"


class StubDetector(ObjectDetector):
    def detect(self, image):
        return [Object(SCENE_TYPE, "office", None), Object("stub", "chair", 0.9)], \
               [Bounds(0, 0, 1, 1), Bounds(0, 0, 0.5, 0.5)]


class StubSource(ImageSource):
    def capture(self):
        return Image(np.zeros((20, 10, 3), dtype=np.uint8), Bounds(0, 0, 1, 1), None)


class TestObjectRecognitionService(unittest.TestCase):
    def setUp(self):
        self.event_bus = SynchronousEventBus()
        self.service = ObjectRecognitionService("image_topic", "object_topic", StubDetector(),
                                                lambda url=None: StubSource(), self.event_bus, None)

    def tearDown(self):
        self.service.stop()

    def test_container_is_importable(self):
        self.assertTrue(issubclass(ObjectRecognitionContainer, object))
        self.assertIsNone(self.service.app)

    def test_stop_before_start(self):
        self.service.stop()

    def test_object_event_carries_scenario_and_tenant(self):
        self.service.start()
        events = Queue()
        self.event_bus.subscribe("object_topic", events.put)

        signal = ImageSignal.for_scenario(SCENARIO_ID, 0, 1, "cltl-storage:image/1", (0, 0, 10, 20))
        source = Event.for_scenario_payload(SCENARIO_ID, ImageSignalEvent.create(signal))
        source = dataclasses.replace(source, metadata=dataclasses.replace(source.metadata, tenant=TENANT))
        self.event_bus.publish("image_topic", source)

        event = events.get(block=True, timeout=1)
        self.assertIsInstance(event.payload, ObjectRecognitionEvent)
        self.assertEqual(2, len(event.payload.mentions))
        self.assertEqual(SCENARIO_ID, event.metadata.scenario_id)
        self.assertEqual(TENANT, event.metadata.tenant)
