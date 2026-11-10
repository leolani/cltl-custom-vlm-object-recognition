from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()

setup(
    name='cltl.vlm_object_recognition',
    version=version,
    package_dir={'': 'src'},
    # Only `myorg.*` — never `myorg` itself. That is what leaves `myorg` a PEP
    # 420 namespace package with no __init__.py content of its own, so a second
    # distribution (myorg.other_module) can share the same top-level namespace
    # without either one shadowing the other. See docs/component.md.
    packages=find_namespace_packages(include=['cltl.vlm_object_recognition','cltl.*'], where='src'),
    data_files=[('VERSION', ['VERSION'])],
    url="https://github.com/leolani/cltl-vlm-object-recognition",
    license='MIT License',
    author='Your organisation',
    author_email='you@example.org',
    description='Template module for attaching custom processing to a Leolani deployment',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.10',
    # Deliberately unpinned, following every other component in this platform:
    # resolution is entirely positional against whatever sdist currently sits in
    # cltl-requirements/leolani/, selected by `--pre --upgrade --upgrade-strategy
    # eager --no-index` (see makefile / util/make/makefile.py.base.mk).
    #
    # cltl.backend[impl] is here for ONE class: ClientImageSource, which
    # resolves the `cltl-storage:image/<id>` reference an image signal carries
    # and decodes the pixels out of the storage service's JSON-with-base64 wire
    # format (see myorg/example/service.py). The `impl` extra adds only mock,
    # requests and parameterized on top of cltl.backend's own numpy — all pure
    # Python, and all but `requests` already preinstalled in the cltl-base
    # image; see docs/docker.md for the caveat about that one. A module that
    # leaves [myorg.example] topic_image empty never imports any of it, but the
    # dependency is declared rather than optional because the shipped
    # configuration DOES use the image path.
    install_requires=['cltl.combot', 'emissor', 'cltl.backend[impl]'],
    extras_require={
        "service": [
            # cltl.combot.infra.container imports the kombu event bus at module
            # level, so every component that extends InfraContainer needs this
            # extra whether or not the deployment actually uses kombu. It is an
            # extra of cltl.combot rather than a dependency of this component,
            # hence here and not in install_requires.
            "cltl.combot[external]",
            # cltl-eliza gets Flask/Werkzeug transitively through
            # cltl.emissor-data's own `service` extra — this template has no
            # such dependency to piggyback on, so src/main.py's `/health`
            # endpoint (needed by the Dockerfile's HEALTHCHECK) needs them
            # declared directly.
            "flask",
            "werkzeug",
        ]}
)
