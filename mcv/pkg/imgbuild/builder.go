package imgbuild

import (
	"fmt"

	"github.com/redhat-et/MCU/mcv/pkg/utils"
	logging "github.com/sirupsen/logrus"
)

type ImageBuilder interface {
	CreateImage(imgName string, cacheDir string) error
}

var HasApp = utils.HasApp

func New() (ImageBuilder, error) {
	if HasApp("buildah") {
		logging.Infof("Using buildah to build the image")
		return &buildahBuilder{}, nil
	} else if HasApp("docker") {
		logging.Infof("Using docker to build the image")
		return &dockerBuilder{}, nil
	}
	return nil, fmt.Errorf("unsupported builder: neither buildah nor docker found")
}

func NewWithBuilder(builder string) (ImageBuilder, error) {
	switch builder {
	case "buildah":
		if HasApp("buildah") {
			logging.Infof("Using buildah to build the image")
			return &buildahBuilder{}, nil
		}
		return nil, fmt.Errorf("buildah is not available on this system")
	case "docker":
		if HasApp("docker") {
			logging.Infof("Using docker to build the image")
			return &dockerBuilder{}, nil
		}
		return nil, fmt.Errorf("docker is not available on this system")
	case "":
		return New() // Fallback to auto-detection
	default:
		return nil, fmt.Errorf("unsupported builder: %s", builder)
	}
}
