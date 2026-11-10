import abc
import dataclasses
from typing import Iterable, Tuple, Optional

import numpy as np
from cltl.backend.api.camera import Bounds


@dataclasses.dataclass
class Object:
    """
    Information about an object.

    Includes the type of the object
    """
    type: str
    label: str
    confidence: Optional[float]


# Object.type of the whole-image scene classification and scene description, as opposed to
# individual object detections, which use the name of the detection model as type.
SCENE_TYPE = "scene"
SCENE_DESCRIPTION_TYPE = "scene_description"


class ObjectDetector(abc.ABC):
    """
    Detect objects in an image.
    """

    def detect(self, image: np.ndarray) -> Tuple[Iterable[Object], Iterable[Bounds]]:
        """
        Detect objects in an image.

        Parameters
        ----------
        image : np.ndarray
            The binary image.

        Returns
        -------
        Iterable[Object]
            The objects detected in the image.
        Iterable[Bounds]
            The bounding boxes of the detected objects in the image. Provides
            the same number of Bounds in the same order as the returned objects.
        """
        raise NotImplementedError()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
