APP_PORT := 5021
DOCKER_TAG := latest
DOCKER_IMAGE := service

DEPLOY_HOST := demo_host

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: build
build:
	docker build -f Dockerfile . --force-rm=true -t $(DOCKER_IMAGE):$(DOCKER_TAG)
