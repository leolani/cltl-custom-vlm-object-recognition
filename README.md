# cltl-vlm-object-recognition

A Leolani module that **recognises objects and scenes in images** with a
vision-language model (VLM), for example Qwen3-VL served by llama.cpp's
`llama-server`. It consumes the image events and
produces object annotations, plus a scene label and a
one-sentence description of the image.

For every image it reports:

- **scene**: a short label for the place, e.g. `office`, `kitchen`, `street`
- **scene description**: one sentence on what is going on in the image
- **objects**: up to 40 objects, each with a label, a bounding box and,
  if the model gives one, a confidence score

## How it fits in the platform

Leolani components don't call each other. They publish and subscribe to
*topics* on a shared message bus (RabbitMQ). This module subscribes to image
events and publishes object annotations. Nothing in the platform has to be
changed or told that the module exists.

```
  chat UI ── upload picture ──► storage service (cltl-storage:image/<id>)
     │
     │ publishes ImageSignalEvent on  cltl.topic.image
 ════▼═══════════════════════════════════════════════════  bus
                    │
          ┌─────────▼───────────┐   GET image   ┌─────────────────┐
          │ vlm-object-         │ ◄──────────── │ storage service │
          │ recognition         │               └─────────────────┘
          │                     │   prompt+img  ┌─────────────────┐
          │                     │ ────────────► │  llama-server   │
          └─────────┬───────────┘ ◄──── JSON ── │  (Qwen3-VL)     │
                    │                           └─────────────────┘
                    │ publishes ObjectRecognitionEvent on
 ═══════════════════▼════════════════════════════  cltl.topic.object_recognition
                    │
          downstream modules (e.g. monitoring, EMISSOR, reasoners)
```

## Input

**Topic:** `[cltl.object_recognition.events] image_topic`, default
`cltl.topic.image`.

**Payload:** an `ImageSignalEvent`. The event carries no pixels. Its
`signal.files[0]` is a reference such as `cltl-storage:image/<id>`. The
module resolves the reference against `[cltl.backend] storage_url` and
fetches the image over HTTP from the storage service, using `cltl.backend`'s
`ClientImageSource`.

The image is scaled down to at most 1024 px on its longest side, encoded as
PNG and sent to the VLM. The VLM's answer is limited to a JSON schema, so the
reply always has the shape `{scene, scene_description, objects[]}`.

## Output

**Topic:** `[cltl.object_recognition.events] object_topic`, default
`cltl.topic.object_recognition`.

**Payload:** an `ObjectRecognitionEvent`, which is an EMISSOR
`AnnotationEvent`. It holds one `Mention` per result. Each mention has:

- a **segment** on the original image signal: the object's bounding box in
  pixel coordinates `(x0, y0, x1, y1)`, or the whole image for the scene and
  the description
- one **annotation** whose value is an `Object(type, label, confidence)`:

| `Object.type` | `label` | Segment |
|---|---|---|
| `scene` | short scene label, e.g. `kitchen` | whole image |
| `scene_description` | one-sentence description | whole image |
| the model name, e.g. `Qwen3VL-8B-Instruct` | object label, e.g. `cup` | object's bounding box |

If nothing is detected, the event holds a single mention over the whole image
with an annotation value of `None`.

The output event is published with `source=` the input event, so it keeps
the incoming event's **scenario id and tenant**. That is what routes the
result back to the correct conversation and tenant.

## Requirements

1. **A running Leolani deployment**: the event bus (RabbitMQ), the backend's
   storage service, the tenant's chat UI and `context` module. See the
   `leolani_server` / `leolani_chatui_tenant` checkouts and their
   `doc/DEPLOYMENT.md`.
2. **A llama-server with a vision model**, for example:

   ```bash
   llama-server -m Qwen3VL-8B-Instruct-Q4_K_M.gguf \
                --mmproj mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf \
                -c 4092 -ngl all --host 0.0.0.0 --port 9009
   ```

   If the log warns that the response was truncated, raise the context size
   (`-c`).
3. **Python 3.10+** and the dependencies:

   ```bash
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Running as a tenant module

In a multi-tenant deployment, one shared server runs the event bus and the
shared modules, and each tenant (user group) has its own chat UI and context.
Tenants are kept apart **only** by the routing key: a tenant module binds
`cltl.topic.image.tenant-a` and publishes on
`cltl.topic.object_recognition.tenant-a`. To serve one tenant, set
`CLTL_TENANT` to that tenant's id.

`src/main.py` reads `config/default.config` with `config/custom.config`
layered on top. All of this module's own settings are in `custom.config`:

```ini
[cltl.event]
implementation: kombu

[cltl.event.kombu]
server: $CLTL_AMQP_URL
exchange: cltl.combot
compression: bzip2
tenant: $CLTL_TENANT

[cltl.backend]
storage_url: $CLTL_STORAGE_URL           # where cltl-storage:image/<id> resolves
server_image_url: http://127.0.0.1:8000/host

[cltl.object_recognition]
model: Qwen3VL-8B-Instruct               # only used as Object.type / in logs
host: http://localhost:9009              # llama-server address
frequency_penalty: 0.3                   # stops the model repeating detections
# api_key: ...                           # only if llama-server uses --api-key

[cltl.object_recognition.events]
image_topic: cltl.topic.image
object_topic: cltl.topic.object_recognition
```

Then start it with the three environment variables set:

```bash
CLTL_AMQP_URL=amqp://eliza:eliza123@127.0.0.1:5672/ \
CLTL_TENANT=tenant-a \
CLTL_STORAGE_URL=http://127.0.0.1:8001/storage/ \
.venv/bin/python src/main.py
```

| Variable | Meaning |
|---|---|
| `CLTL_AMQP_URL` | RabbitMQ address of the shared server, as seen from where `main.py` runs |
| `CLTL_TENANT` | The tenant this module serves. If empty, it subscribes to `cltl.topic.image.#` and answers **every** tenant, which is only correct for a single-tenant deployment |
| `CLTL_STORAGE_URL` | The storage service that holds the tenant's uploaded images. **Must end with `/`**. Port 8001 in a standalone/server deployment; in the split layout a tenant's own backend runs on 9001 |

The service **refuses to start** if `storage_url` is empty or still the
literal `$CLTL_STORAGE_URL` (the config reader does not fail on unset
variables, it passes the text through). Without that check, the first image
fails with `requests.exceptions.InvalidURL: No host specified`. To check the
URL, take an image id from the log and run:

```bash
curl -I http://127.0.0.1:8001/storage/image/<id>      # expect 200
```

When it is running, open the tenant's chat UI, upload a picture in the Image
tab, and the module logs `Detected scene '...' and objects: [...]` and
publishes the annotations. A `/health` endpoint is served on port 8010.

Scenarios are opened by the platform's `context` module, not by this module.
It annotates images in whichever scenario is already open.

## Trying the detector without the platform

The VLM part can be run on a local image file, with no bus or storage:

```bash
python -m cltl.vlm_object_recognition.llama_cpp image-1.jpeg --host http://localhost:9009
```

It prints the scene, the description and the detections, and writes an
annotated copy to `image-1.detections.png` (`--show` also opens it in a
window).

## Code layout

| File | Role |
|---|---|
| `src/cltl/vlm_object_recognition/api.py` | `Object` and the `ObjectDetector` interface |
| `src/cltl/vlm_object_recognition/llama_cpp.py` | `LlamaCppObjectDetector`: prompt, JSON schema, bounding-box conversion; CLI |
| `src/cltl/vlm_object_recognition/service.py` | `ObjectRecognitionService`: subscribes, fetches the image, detects, publishes |
| `src/cltl/vlm_object_recognition/schema.py` | `ObjectRecognitionEvent`: turns detections into EMISSOR mentions |
| `src/cltl/vlm_object_recognition/container.py` | Wires detector and service into the platform's container |
| `src/main.py` | Entry point: loads config, starts the container and the `/health` server |

## License

MIT, see [`LICENSE`](LICENSE).
