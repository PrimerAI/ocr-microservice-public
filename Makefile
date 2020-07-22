PROJECT=ocr-microservice
SHELL=/bin/bash
IMAGE_BASE=ocr-microservice

help:  ## Display this help text
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean:  ## Remove build, cache, pyc files and build artifacts
	rm -rf build/ dist/ .eggs/ ${PROJECT}.egg-info/ .coverage*
	find . -iname '*.pyc' -o -iname '__pycache__' | xargs rm -rf
	docker-compose -p ${PROJECT} rm -fsv || true

checks:
	@ [ -f ~/.pip/pip.conf ] || (echo "Missing ~/.pip/pip.conf - Get from 1Password";)
	@ (which aws &>/dev/null || (echo "Missing aws command"; exit 1))

ensure-venv:
	@ [ \( "x${VIRTUAL_ENV}" != "x" \) -o \( "x${CI}" != "x" \) ] || ( echo 'Please activate your venv using "workon ${PROJECT}"'; exit 1 )

# =========
# = Setup =
# =========

setup-local: checks  ## Setup this repo's local requirements (brew, virtualenv requirements…)
	which aws &>/dev/null || brew install awscli
	which realpath &>/dev/null || brew install coreutils
	[ -d /Applications/Docker.app ] || brew cask install docker
	which docker &>/dev/null || brew install docker
	which docker-compose &>/dev/null || brew install docker-compose
	which jq &>/dev/null || brew install jq
	(gcc -lxmlsec1 2>&1 | grep -q "library not found") && brew install libxmlsec1 || true
	which nvm &>/dev/null || brew ls nvm || brew install nvm
	which yarn &>/dev/null || brew install yarn
	which java &>/dev/null || brew cask install java
	which nvm &>/dev/null || source /usr/local/opt/nvm/nvm.sh &>/dev/null; nvm install lts/*
	which python3.8 &>/dev/null || brew install python@3.8
	pip install virtualenv virtualenvwrapper
	( \
		which workon || source /usr/local/bin/virtualenvwrapper.sh; \
		workon ${PROJECT} || mkvirtualenv -p python3.8 ${PROJECT}; \
		$(MAKE) pip-install \
	)

pip-clean: ensure-venv  ## Uninstall all pip packages in the current env
	python3 -m pip uninstall -y -r <(python3 -m pip freeze)

pip-install: ensure-venv  ## Install test requirements
	python3 -m pip install --upgrade pip==20.1.1 setuptools==47.1.1
	python3 -m pip install -r requirements-dev.txt

# =====================
# = Docker, local run =
# =====================

docker-start:
	@ docker-compose -p ${PROJECT} up -d

docker-stop:
	@ docker-compose -p ${PROJECT} rm -fsv
	@ docker network rm $$(docker network list -f name=${PROJECT} -q) || true # ignore errors until docker cli has -f option: https://github.com/docker/cli/issues/2382

docker-restart: docker-stop docker-start

docker-build: ensure-venv ## Build docker container
	@ ( \
			export GITHUB_SHA=$${GITHUB_SHA:-dev} && \
			[[ "x${CI}" == "x" ]] || export CI_DOCKER_OPTS="--progress=plain" && \
			DOCKER_BUILDKIT=1 docker build $$CI_DOCKER_OPTS \
				--build-arg BUILDKIT_INLINE_CACHE=1 \
				--secret id=pip_extra_index_url,src=$$PIP_CONFIG_FILE \
				--cache-from 963188529772.dkr.ecr.us-west-2.amazonaws.com/${IMAGE_BASE}:latest \
				-t ${IMAGE_BASE} -t ${IMAGE_BASE}:$${GITHUB_SHA:0:7} \
				. \
	)

user-app-local: checks ensure-venv docker-start  ## Run app locally
	@ ACCEPTED_DOMAINS=.* PYTHONPATH=$(shell pwd):$PYTHONPATH ./scripts/env.sh python3 ${PROJECT}/run.py


# TODO (russell): add line-length=100
run-black:
	@ python3 -m black --quiet $$BLACK_ARGS

lint-start:
	@ echo "Running all lints…"

lint-black:
	@ echo "    … black"
	@ BLACK_ARGS="--fast --check ." $(MAKE) run-black || (echo "Please run 'make tidy'" && exit 1)

lint-flake8:
	@ echo "    … flake8"
	@ python3 -m flake8 ${PROJECT} $$FLAKE8_ARGS
	@ python3 -m flake8 *tests --ignore=E203,F401,F811,I100,I101,I201,I202,W503 $$FLAKE8_ARGS


lint-pylint:
	@ echo "    … pylint"
	@ ( \
		[ -z $$PYLINT_OUTPUT_FILE ] || exec 3>&1 1>$$PYLINT_OUTPUT_FILE; \
		python3 -m pylint --rcfile=setup.cfg ${PROJECT} $$PYLINT_ARGS || exit 1; \
		python3 -m pylint --rcfile=setup.cfg \
			*tests $$PYLINT_ARGS || exit 1; \
		[ -z $$PYLINT_OUTPUT_FILE ] || exec 1>&3; \
	)

LINTS := $(patsubst %,lint-%,start black flake8 pylint)
lint: $(LINTS)  ## Run linting checks (individual `lint-%` commands also exist, e.g. `lint-flake8`)

git-new-and-modified:
	@ git diff ${BASE_BRANCH} --diff-filter=ACMR --name-only | (grep -E '.py$$' || true) | tr '\n' ' '

tidy:  ## Fix linting for modified files
	@ ( \
		export MODIFIED="$$(BASE_BRANCH=origin/master $(MAKE) -s git-new-and-modified)" && \
		if [ -z "$${MODIFIED}" ]; then echo "No files changed – no need to tidy"; exit; fi && \
		echo "Tidying python files:\n\n$${MODIFIED}\n" && \
		$(MAKE) tidy-all \
	)

tidy-all:
	@ black --quiet $${MODIFIED:-.}


# =========
# = Tests =
# =========

test-local: checks ensure-venv clean docker-start test-local-no-clean  ## Run unit tests

test-integration: checks ensure-venv clean docker-start test-integration-no-clean  ## Run integration tests

tests: test
test: test-all
test-all: test-local test-integration  ## Run all tests

test-local-no-clean: checks ensure-venv  ## Run unit tests
	./scripts/env.sh python3 -m pytest --durations=15 --cov=${PROJECT} --cov-report term-missing --tb=long -v --doctest-modules $$PYTEST_ARGS tests || exit 1 ; \

test-integration-no-clean: checks ensure-venv docker-start  ## Run integration tests
	./scripts/env.sh python3 -m pytest --durations=15 --cov=${PROJECT} --cov-append --cov-report term-missing --tb=long -v --doctest-modules $$PYTEST_ARGS integration_tests || exit 1 ; \

tests-no-clean: test-no-clean
test-no-clean: test-all-no-clean
test-all-no-clean: test-local-no-clean test-integration-no-clean  ## Run all tests

test-args: checks ensure-venv docker-start
	./scripts/env.sh python3 -m pytest --durations=15 --cov=${PROJECT} --cov-report term-missing --tb=long -v --doctest-modules $$PYTEST_ARGS || exit 1 ; \

# ==============
# = Migrations =
# ==============

migration: ensure-venv  ## Create new migration
	./scripts/env.sh python3 -m alembic -c opt/config/alembic.ini revision --autogenerate -m "$$NAME"

migrate: ensure-venv  ## Run migrations
	./scripts/env.sh python3 -m alembic -c opt/config/alembic.ini upgrade head
