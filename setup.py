from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()

setup(
    name='cltl.vlm_object_recognition',
    version=version,
    package_dir={'': 'src'},
    packages=find_namespace_packages(include=['cltl.*', 'cltl_service.*'], where='src'),
    data_files=[('VERSION', ['VERSION'])],
    url="https://github.com/leolani/cltl-vlm-object-recognition",
    license='MIT License',
    author='CLTL',
    author_email='t.baier@vu.nl',
    description='Object and scene recognition with a vision-language model for the Leolani platform',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.10',
    # cltl.backend[impl] is here for ONE class: ClientImageSource, which
    # resolves the `cltl-storage:image/<id>` reference an image signal carries
    # (see cltl_service/vlm_object_recognition/service.py). It imports
    # `requests` at module level, which cltl.backend declares only in that extra.
    # cv2 is deliberately not named: it is imported inside a try.
    install_requires=['cltl.combot', 'emissor', 'cltl.backend[impl]', 'openai', 'numpy'],
    extras_require={
        "service": [
            # cltl.combot.infra.container imports the kombu event bus at module
            # level, so every component that extends InfraContainer needs this
            # extra whether or not the deployment actually uses kombu. It is an
            # extra of cltl.combot rather than a dependency of this component,
            # hence here and not in install_requires.
            "cltl.combot[external]",
            # src/main.py's `/health` endpoint (the Dockerfile's HEALTHCHECK)
            # needs Flask/Werkzeug.
            "flask",
            "werkzeug",
        ]}
)
