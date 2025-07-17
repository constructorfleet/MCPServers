.PHONY: all build-base build-kagi build-plex push-base push-kagi push-plex push

REGISTRY ?= docker.home.prettybaked.com
ORG ?= constructorfleet

BASE_IMAGE := $(REGISTRY)/$(ORG)/mcp-base
KAGI_IMAGE := $(REGISTRY)/$(ORG)/mcp-kagi
PLEX_IMAGE := $(REGISTRY)/$(ORG)/mcp-plex

all: build-base build-kagi build-plex

build-base:
	@BASE_VERSION=$$(grep '^version =' base/pyproject.toml | cut -d'"' -f2); \
	echo "🚧 Building base: $(BASE_IMAGE):$$BASE_VERSION"; \
	docker build -f base/Dockerfile \
		-t $(BASE_IMAGE):$$BASE_VERSION \
		-t $(BASE_IMAGE):latest .

build-kagi: build-base
	@BASE_VERSION=$$(grep '^version =' base/pyproject.toml | cut -d'"' -f2); \
	KAGI_VERSION=$$(grep '^version =' kagi/pyproject.toml | cut -d'"' -f2); \
	echo "🧠 Building kagi: $(KAGI_IMAGE):$$KAGI_VERSION"; \
	docker build -f kagi/Dockerfile \
		--build-arg BASE_VERSION=$$BASE_VERSION \
		-t $(KAGI_IMAGE):$$KAGI_VERSION \
		-t $(KAGI_IMAGE):latest .

build-plex: build-base
	@BASE_VERSION=$$(grep '^version =' base/pyproject.toml | cut -d'"' -f2); \
	PLEX_VERSION=$$(grep '^version =' plex/pyproject.toml | cut -d'"' -f2); \
	echo "🎛 Building plex: $(PLEX_IMAGE):$$PLEX_VERSION"; \
	docker build -f plex/Dockerfile \
		--build-arg BASE_VERSION=$$BASE_VERSION \
		-t $(PLEX_IMAGE):$$PLEX_VERSION \
		-t $(PLEX_IMAGE):latest .

push-base:
	@BASE_VERSION=$$(grep '^version =' base/pyproject.toml | cut -d'"' -f2); \
	docker push $(BASE_IMAGE):$$BASE_VERSION

push-kagi:
	@KAGI_VERSION=$$(grep '^version =' kagi/pyproject.toml | cut -d'"' -f2); \
	docker push $(KAGI_IMAGE):$$KAGI_VERSION

push-plex:
	@PLEX_VERSION=$$(grep '^version =' plex/pyproject.toml | cut -d'"' -f2); \
	docker push $(PLEX_IMAGE):$$PLEX_VERSION

push: push-base push-kagi push-plex