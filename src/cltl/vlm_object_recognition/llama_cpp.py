import base64
import json
import logging
import re
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from cltl.backend.api.camera import Bounds
from openai import OpenAI

from cltl.vlm_object_recognition.api import Object, ObjectDetector, SCENE_TYPE, SCENE_DESCRIPTION_TYPE

logger = logging.getLogger(__name__)


# llama-server serves a single model and ignores the model name in requests,
# it is only used for logging and as Object.type of detections.
_DEFAULT_MODEL = "Qwen3VL-8B-Instruct"
_DEFAULT_HOST = "http://localhost:9009"

# Object.type used for the whole-image scene classification, as opposed to individual
# object detections, which use the model name (see LlamaCppObjectDetectorProxy._to_object).
_SCENE_TYPE = SCENE_TYPE

# The prompt and the response must fit in the context size of llama-server (-c). Image
# tokens scale with the image size (Qwen3-VL: one token per 32x32 pixels), so images are
# downscaled before they are sent, and the number of reported objects is limited.
_MAX_IMAGE_SIZE = 1024
_MAX_OBJECTS = 10
# Penalizes tokens by how often they already occurred, to keep the model from looping on the
# same detection (e.g. dozens of identical "cup" entries). Kept low, as it also applies to the
# JSON keys and coordinate digits that legitimately repeat in the response.
_DEFAULT_FREQUENCY_PENALTY = 0.3

_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {
            "type": "string",
            "description": "A short label classifying the overall scene or place depicted in the "
                            "image, e.g. 'office', 'kitchen', 'living room', 'street', 'city', "
                            "'village', 'forest'.",
        },
        "scene_description": {
            "type": "string",
            "description": "One sentence describing what is going on in the image.",
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["label", "bbox_2d"],
            },
            "maxItems": _MAX_OBJECTS,
        }
    },
    "required": ["scene", "scene_description", "objects"],
}

_PROMPT = (
    "Analyze this image. First classify the overall scene or place it depicts with a short "
    "label, e.g. \"office\", \"kitchen\", \"living room\", \"street\", \"city\", \"village\", "
    "\"forest\", and report it as \"scene\". "
    "Describe in one sentence what is going on in the image and report it as \"scene_description\". "
    "Then detect every distinct object in the image. For each object report its label "
    "and a bounding box as \"bbox_2d\": [xmin, ymin, xmax, ymax], with coordinates "
    "normalized to the range 0-1000 relative to the image height and width. "
    "If you can estimate your confidence, add a \"confidence\" value between 0 and 1. "
    f"Report only objects you can actually see, at most {_MAX_OBJECTS}, the most salient first. "
    "Answer with compact JSON on a single line, without indentation."
)


class LlamaCppObjectDetector(ObjectDetector):
    """
    ObjectDetector implementation that uses a vision-language model (e.g. Qwen3-VL)
    served through llama.cpp's llama-server, e.g.

        llama-server -m Qwen3VL-8B-Instruct-Q4_K_M.gguf --mmproj mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf \\
            -c 4092 -ngl all --host 0.0.0.0 --port 9009

    instead of a dedicated object detection model like Yolo5. llama-server exposes an
    OpenAI compatible API, which is accessed through the OpenAI client.

    The model is prompted to return detections with normalized bounding boxes as
    structured JSON, which is parsed into the same Object/Bounds shape produced by
    ObjectDetectorProxy, so this can be used as a drop-in alternative.

    In addition to individual objects, the model is asked to classify the overall
    scene depicted (e.g. "office", "kitchen", "street"). This is returned as an
    additional Object with type SCENE_TYPE, whose Bounds cover the complete image. A one
    sentence description of the image is returned likewise with type SCENE_DESCRIPTION_TYPE.
    """

    @classmethod
    def from_config(cls, config_manager):
        config = config_manager.get_config("cltl.object_recognition")

        return cls(model=config.get("model") if "model" in config else _DEFAULT_MODEL,
                   host=config.get("host") if "host" in config else _DEFAULT_HOST,
                   api_key=config.get("api_key") if "api_key" in config else None,
                   frequency_penalty=config.get_float("frequency_penalty")
                       if "frequency_penalty" in config else _DEFAULT_FREQUENCY_PENALTY)

    def __init__(self, model: str = _DEFAULT_MODEL, host: str = _DEFAULT_HOST, api_key: str = None,
                 frequency_penalty: float = _DEFAULT_FREQUENCY_PENALTY):
        """
        Parameters
        ----------
        model : str
            Name of the model. llama-server serves a single model and ignores this,
            it is used to label the detections.
        host : str
            Address of the llama-server, e.g. "http://localhost:9009".
        api_key : str
            API key, only needed if llama-server was started with --api-key.
        frequency_penalty : float
            Penalty on repeated tokens, to prevent the model from repeating the same detection.
        """
        self._client = OpenAI(base_url=f"{host.rstrip('/')}/v1", api_key=api_key or "no-key")
        self._model = model
        self._frequency_penalty = frequency_penalty

    def detect(self, image: np.ndarray) -> Tuple[Iterable[Object], Iterable[Bounds]]:
        logger.info("Processing image %s with model %s", image.shape, self._model)

        height, width = image.shape[:2]
        result = self._detect(image)

        objects = []
        bounds = []

        scene = result.get("scene")
        if scene:
            objects.append(Object(_SCENE_TYPE, scene, None))
            bounds.append(Bounds(0, width, 0, height))

        scene_description = result.get("scene_description")
        if scene_description:
            objects.append(Object(SCENE_DESCRIPTION_TYPE, scene_description, None))
            bounds.append(Bounds(0, width, 0, height))

        for detection in result.get("objects", ()):
            parsed = self._to_object(detection, width, height)
            if parsed is None:
                continue
            obj, bound = parsed
            objects.append(obj)
            bounds.append(bound)

        logger.info("Detected scene '%s' and objects: %s", scene, [obj.label for obj in objects if obj.type not in (SCENE_TYPE, SCENE_DESCRIPTION_TYPE)])

        return tuple(objects), tuple(bounds)

    def _detect(self, image: np.ndarray) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._to_data_url(image)}},
                    {"type": "text", "text": _PROMPT},
                ],
            }],
            # llama-server converts the JSON schema to a grammar that constrains the output
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "detections", "schema": _DETECTION_SCHEMA},
            },
            temperature=0.0,
            frequency_penalty=self._frequency_penalty,
        )

        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning("Response was truncated, consider increasing the context size (-c) of llama-server")

        content = choice.message.content or ""
        return self._parse_response(content)

    def _parse_response(self, content: str) -> dict:
        # Reasoning/"thinking" models may emit <think>...</think> reasoning into
        # the content, so strip it out before parsing as JSON.
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # As a last resort, look for the first top-level JSON object in the text,
        # in case the model added commentary around it despite the schema constraint.
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse detection response as JSON: %s", content)
        return {}

    def _to_object(self, detection: dict, width: int, height: int) -> Optional[Tuple[Object, Bounds]]:
        label = detection.get("label")
        box = detection.get("bbox_2d")
        if not label or not box or len(box) != 4:
            logger.warning("Skipping malformed detection: %s", detection)
            return None

        # Qwen-VL models are trained on [xmin, ymin, xmax, ymax] boxes normalized to 0-1000
        xmin, ymin, xmax, ymax = box
        x0 = max(0.0, min(width, xmin / 1000 * width))
        x1 = max(0.0, min(width, xmax / 1000 * width))
        y0 = max(0.0, min(height, ymin / 1000 * height))
        y1 = max(0.0, min(height, ymax / 1000 * height))
        if x1 <= x0 or y1 <= y0:
            logger.warning("Skipping detection with empty bounding box: %s", detection)
            return None

        return Object(self._model, label, detection.get("confidence")), Bounds(x0, x1, y0, y1)

    def _to_data_url(self, image: np.ndarray) -> str:
        # Bounding boxes are normalized, so downscaling does not affect them
        scale = _MAX_IMAGE_SIZE / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        is_success, buffer = cv2.imencode(".png", image)

        if not is_success:
            raise ValueError("Could not encode image")

        return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")

# python -m cltl.vlm_object_recognition.llama_cpp  <image_path> [--model ...] [--host ...] [--api-key ...] [--show]
# python -m cltl.vlm_object_recognition.llama_cpp image-1.jpeg --host http://localhost:9009
def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Detect objects in an image using a VLM served through llama-server.")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Model name (ignored by llama-server, used to label detections)")
    parser.add_argument("--host", default=_DEFAULT_HOST, help="llama-server address")
    parser.add_argument("--api-key", default=None, help="API key, if llama-server was started with --api-key")
    parser.add_argument("--output", default=None,
                         help="Where to save the annotated image (default: <image>.detections.png)")
    parser.add_argument("--show", action="store_true",
                         help="Also open the annotated image in a window (blocks until a key is pressed)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")

    proxy = LlamaCppObjectDetector(model=args.model, host=args.host, api_key=args.api_key)
    objects, bounds = proxy.detect(image)

    found_objects = False
    for obj, bound in zip(objects, bounds):
        if obj.type == _SCENE_TYPE:
            print(f"Scene: {obj.label}")
            cv2.putText(image, f"scene: {obj.label}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            continue
        if obj.type == SCENE_DESCRIPTION_TYPE:
            print(f"Description: {obj.label}")
            continue

        found_objects = True
        print(f"{obj.label} (confidence={obj.confidence}): {bound}")
        cv2.rectangle(image, (int(bound.x0), int(bound.y0)), (int(bound.x1), int(bound.y1)), (0, 255, 0), 2)
        cv2.putText(image, obj.label, (int(bound.x0), max(0, int(bound.y0) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if not found_objects:
        print("No objects detected")

    output_path = args.output or f"{os.path.splitext(args.image)[0]}.detections.png"
    cv2.imwrite(output_path, image)
    print(f"Annotated image written to {output_path}")

    if args.show:
        cv2.imshow("Detections", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
