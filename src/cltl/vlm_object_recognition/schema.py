import logging
import uuid
from dataclasses import dataclass
from typing import Iterable, Tuple

from cltl.backend.api.camera import Bounds
from cltl.combot.event.emissor import AnnotationEvent
from cltl.combot.infra.time_util import timestamp_now
from emissor.representation.container import MultiIndex
from emissor.representation.scenario import Mention, ImageSignal, Annotation, module_source, class_type

from cltl.vlm_object_recognition.api import Object

logger = logging.getLogger(__name__)

@dataclass
class ObjectRecognitionEvent(AnnotationEvent[Annotation[Object]]):
    @classmethod
    def create_obj_rec_event(cls, image_signal: ImageSignal, objects: Iterable[Object], bounds: Iterable[Bounds],
                             image_size: Tuple[int, int] = None):
        """
        image_size : (width, height) of the processed image in pixels. Used as the image
        bounds if the signal's bounds are unknown, e.g. the backend's (0, 0, -1, -1) fallback
        for cameras that don't report their resolution.
        """
        if objects:
            mentions = [ObjectRecognitionEvent.to_mention(image_signal, object, bound, image_size)
                        for object, bound in zip(objects, bounds)]
        else:
            mentions = [ObjectRecognitionEvent.to_mention(image_signal, image_size=image_size)]

        return cls(cls.__name__, mentions)

    @staticmethod
    def to_mention(image_signal: ImageSignal, object: Object = None, bounds: Bounds = None,
                   image_size: Tuple[int, int] = None):
        """
        Create Mention with object annotations. If no face is detected, annotate the whole
        image with Object Annotation with value None.
        """
        segment = image_signal.ruler
        x0, y0, x1, y1 = segment.bounds
        if image_size and (x1 <= x0 or y1 <= y0):
            segment = MultiIndex(segment.container_id, (0, 0, int(image_size[0]), int(image_size[1])))
        if bounds:
            clipped = Bounds.from_diagonal(*segment.bounds).intersection(bounds)
            if clipped:
                # Emissor segments use integer pixel coordinates
                segment = segment.get_area_bounding_box(int(clipped.x0), int(clipped.y0),
                                                        int(clipped.x1), int(clipped.y1))
            else:
                # The signal's bounds don't match the image's pixel size (e.g. the
                # backend's (-1, -1) fallback), so annotate the whole image instead.
                logger.warning("Object bounds %s outside image signal bounds %s, annotating whole image",
                               bounds, segment.bounds)

        annotation = Annotation(class_type(object), object, module_source(__name__), timestamp_now())

        return Mention(str(uuid.uuid4()), [segment], [annotation])


if __name__ == '__main__':
    from emissor.representation.util import marshal
    signal = ImageSignal.for_scenario("sc_id1", 0, 1, "", (0,0,1,1))
    event = ObjectRecognitionEvent.create_obj_rec_event(signal, [Object("chair")], [Bounds(0,1,0,1)])
    print(marshal(event))