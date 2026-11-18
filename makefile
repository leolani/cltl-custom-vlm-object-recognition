SHELL = /bin/bash

project_dependencies ?= $(addprefix $(project_root)/, \
		emissor \
		cltl-requirements \
		cltl-combot \
		cltl-backend)


git_remote ?= https://github.com/leolani


include util/make/makefile.base.mk
include util/make/makefile.component.mk
include util/make/makefile.py.base.mk
include util/make/makefile.git.mk
include util/make/makefile.docker.mk


clean: py-clean
