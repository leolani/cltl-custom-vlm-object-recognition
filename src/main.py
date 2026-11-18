import logging.config
import os
import signal

from cltl.combot.event.emissor import SIG, MEN
from cltl.combot.infra.config.k8config import K8LocalConfigurationContainer
from cltl.combot.infra.di_container import singleton
from cltl.combot.infra.event.api import Event, PAYLOAD
from cltl.combot.infra.event.memory import SynchronousEventBus
from emissor.representation.util import marshal, unmarshal, register_type_var
from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from cltl_service.vlm_object_recognition.container import ObjectRecognitionContainer

# Top-level module, deliberately: find_namespace_packages(include=['cltl.*',
# 'cltl_service.*']) does not match a bare `main`, so this file is never part of
# the installed distribution. Only the container image and a developer running
# it directly ever execute it.

logging.config.fileConfig(os.environ.get('CLTL_LOGGING_CONFIG', 'config/logging.config'),
                          disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# Must happen before anything marshals an Event, or emissor cannot resolve the
# generic type variables and raises `TypeError: PAYLOAD is not a dataclass`.
register_type_var(PAYLOAD)
register_type_var(SIG)
register_type_var(MEN)


def serializer(obj):
    return marshal(obj, cls=Event)


def deserializer(obj):
    return unmarshal(obj, cls=Event)


class ApplicationContainer(ObjectRecognitionContainer):
    """This deployment: just the component. Scenario creation is the platform's
    own cltl-context's job."""

    @property
    @singleton
    def event_bus_serializer(self):
        return serializer, deserializer

    @property
    @singleton
    def event_bus(self):
        config = self.config_manager.get_config("cltl.event")
        if config.get("implementation") == "internal":
            return SynchronousEventBus()
        return super().event_bus


def main():
    K8LocalConfigurationContainer.load_configuration()
    application = ApplicationContainer()

    # `docker compose down` and `docker stop` send SIGTERM, whose DEFAULT
    # disposition terminates the process outright — so `with application:`
    # would never reach its __exit__, the container would never be stopped. Turning the signal into KeyboardInterrupt makes
    # it unwind through the `with` like a Ctrl-C does.
    def _interrupt(signum, frame):
        logger.info("Received signal %s; shutting down", signum)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _interrupt)

    with application:
        flask_app = Flask(__name__)

        @flask_app.route('/health')
        def health():
            return 'OK', 200

        routes = {}
        if application.object_recognition_service and application.object_recognition_service.app:
            routes['/vlm_object_recognition'] = application.object_recognition_service.app

        try:
            run_simple('0.0.0.0', 8010, DispatcherMiddleware(flask_app, routes),
                       threaded=True, use_reloader=False, use_debugger=False)
        except KeyboardInterrupt:
            # Caught here rather than relying on werkzeug: its
            # BaseWSGIServer.serve_forever does swallow KeyboardInterrupt, but
            # that is an implementation detail of a pinned version, and the cost
            # of not depending on it is one except clause.
            pass


if __name__ == '__main__':
    main()
