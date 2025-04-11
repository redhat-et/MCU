all: cargohold

##@ Help

# The help target prints out all targets with their descriptions organized
# beneath their categories. The categories are represented by '##@' and the
# target descriptions by '##'. The awk commands is responsible for reading the
# entire set of makefiles included in this invocation, looking for lines of the
# file as xyz: ## something, and then pretty-format the target and help. Then,
# if there's a line with ##@ something, that gets pretty-printed as a category.
# More info on the usage of ANSI control characters for terminal formatting:
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_parameters
# More info on the awk command:
# http://linuxcommand.org/lc3_adv_awk.php

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

### env define ###
export BIN_TIMESTAMP ?=$(shell date -u +'%Y-%m-%dT%H:%M:%SZ')
export TIMESTAMP ?=$(shell echo $(BIN_TIMESTAMP) | tr -d ':' | tr 'T' '-' | tr -d 'Z')

# restrict included verify-* targets to only process project files
SRC_ROOT           := $(shell pwd)
ARCH               := $(shell arch)
OUTPUT_DIR         := _output
BUILD_BINDIR := $(OUTPUT_DIR)/bin
GIT_VERSION        := $(shell git describe --dirty --tags --always --match='v*')
GIT_SHA            := $(shell git rev-parse HEAD)
GIT_BRANCH         := $(shell git rev-parse --abbrev-ref HEAD)
VERSION            := $(GIT_VERSION)


# set GOENV
GOOS := $(shell go env GOOS)
GOARCH := $(shell go env GOARCH)

GOENV = GO111MODULE="" \
				GOOS=$(GOOS) \
				GOARCH=$(GOARCH)
PKG_BUILD="github.com/tkdk/cargohold/pkg/build"
LDFLAGS := $(LDFLAGS) \
		-X $(PKG_BUILD).Version=$(VERSION) \
		-X $(PKG_BUILD).Revision=$(GIT_SHA) \
		-X $(PKG_BUILD).Branch=$(GIT_BRANCH) \
		-X $(PKG_BUILD).OS=$(GOOS) \
		-X $(PKG_BUILD).Arch=$(GOARCH)

GO_BUILD_TAGS := '$(GOOS)'

ifndef GOPATH
	GOPATH := $(HOME)/go
endif

ifndef GOBIN
	GOBIN := $(GOPATH)/bin
endif

# NOTE: project related tools get installed to tmp dir which is ignored by
PROJECT_DIR := $(shell dirname $(abspath $(firstword $(MAKEFILE_LIST))))

base_dir := $(patsubst %/,%,$(dir $(realpath $(firstword $(MAKEFILE_LIST)))))

##@ Default
cargohold: build ## Build cargohold.
.PHONY: cargohold

##@ General build
build: clean_build_local _build_local  ##  Build binary and copy to $(OUTPUT_DIR)/bin
.PHONY: build

_build_local:  ##  Build CargoHold binary locally.
	@mkdir -p "$(BUILD_BINDIR)/$(GOOS)_$(GOARCH)"
	+@$(GOENV) go build \
		-v -tags ${GO_BUILD_TAGS} \
		-ldflags "$(LDFLAGS)" \
		-o $(BUILD_BINDIR)/$(GOOS)_$(GOARCH)/cargohold \
		./cmd/main.go

## toolkit ###
.PHONY: tidy-vendor
tidy-vendor:
	@echo "******     tidy-vendor     ******"
	@echo
	go mod tidy -v
	go mod vendor
	@echo
	@echo

clean_build_local: ## Clean local build directory
	rm -rf $(BUILD_BINDIR)

format:
	@echo "******     Go Format     ******"
	@echo
	go fmt github.com/tkdk/cargohold/...
	@echo
	@echo
