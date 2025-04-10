package imgbuild

import (
	"fmt"

	logging "github.com/sirupsen/logrus"
	"github.com/tkdk/cargohold/pkg/utils"
)

type ImageBuilder interface {
	CreateImage(imgName string, cacheDir string) error
}

type imgBuilder struct {
	builder ImageBuilder
}

type CacheMetadataWithDummy struct {
	Hash       string `json:"hash"`
	Backend    string `json:"backend"`
	Arch       string `json:"arch"`
	WarpSize   int    `json:"warp_size"`
	PTXVersion *int   `json:"ptx_version,omitempty"`
	DummyKey   string `json:"dummy_key"`
}

// Factory function to create a new ImgBuilder with the specified backend.
func New() (ImageBuilder, error) {
	var builder ImageBuilder
	var builderType string

	if utils.HasApp("buildah") {
		// Favor buildah if it's available
		builderType = "buildah"
	} else if utils.HasApp("docker") {
		builderType = "docker"
	}

	logging.Infof("Using %s to build the image", builderType)

	switch builderType {
	case "docker":
		builder = &dockerBuilder{}
	case "buildah":
		builder = &buildahBuilder{}
	default:
		return nil, fmt.Errorf("unsupported builder type: %s", builderType)
	}

	return &imgBuilder{builder: builder}, nil
}

func (i *imgBuilder) CreateImage(imgName, cacheDir string) error {
	return i.builder.CreateImage(imgName, cacheDir)
}
