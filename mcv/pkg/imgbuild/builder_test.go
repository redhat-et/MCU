package imgbuild

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNew_BuildahAvailable(t *testing.T) {
	origHasApp := HasApp
	defer func() { HasApp = origHasApp }()

	HasApp = func(tool string) bool {
		return tool == "buildah"
	}

	builder, err := New()
	assert.NoError(t, err)
	assert.IsType(t, &buildahBuilder{}, builder)
}

func TestNew_DockerFallback(t *testing.T) {
	origHasApp := HasApp
	defer func() { HasApp = origHasApp }()

	HasApp = func(tool string) bool {
		return tool == "docker"
	}

	builder, err := New()
	assert.NoError(t, err)
	assert.IsType(t, &dockerBuilder{}, builder)
}

func TestNew_Unsupported(t *testing.T) {
	origHasApp := HasApp
	defer func() { HasApp = origHasApp }()

	HasApp = func(tool string) bool {
		return false
	}

	builder, err := New()
	assert.Nil(t, builder)
	assert.Error(t, err)
}

func TestNewWithBuilder_Buildah(t *testing.T) {
	origHasApp := HasApp
	defer func() { HasApp = origHasApp }()

	HasApp = func(tool string) bool {
		return tool == "buildah"
	}

	builder, err := NewWithBuilder("buildah")
	assert.NoError(t, err)
	assert.IsType(t, &buildahBuilder{}, builder)
}

func TestNewWithBuilder_Docker(t *testing.T) {
	origHasApp := HasApp
	defer func() { HasApp = origHasApp }()

	HasApp = func(tool string) bool {
		return tool == "docker"
	}

	builder, err := NewWithBuilder("docker")
	assert.NoError(t, err)
	assert.IsType(t, &dockerBuilder{}, builder)
}

func TestNewWithBuilder_Unsupported(t *testing.T) {
	builder, err := NewWithBuilder("unsupported")
	assert.Nil(t, builder)
	assert.Error(t, err)
}

func TestNewWithBuilder_AutoDetect(t *testing.T) {
	origHasApp := HasApp
	defer func() { HasApp = origHasApp }()

	HasApp = func(tool string) bool {
		return tool == "docker"
	}

	builder, err := NewWithBuilder("")
	assert.NoError(t, err)
	assert.IsType(t, &dockerBuilder{}, builder)
}
