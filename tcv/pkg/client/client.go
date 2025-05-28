package client

import (
	"fmt"
	"os"

	"github.com/redhat-et/TKDK/tcv/pkg/accelerator"
	"github.com/redhat-et/TKDK/tcv/pkg/accelerator/devices"
	"github.com/redhat-et/TKDK/tcv/pkg/config"
	"github.com/redhat-et/TKDK/tcv/pkg/constants"
	"github.com/redhat-et/TKDK/tcv/pkg/fetcher"
	"github.com/redhat-et/TKDK/tcv/pkg/logformat"
	"github.com/redhat-et/TKDK/tcv/pkg/preflightcheck"
	logging "github.com/sirupsen/logrus"
)

type Options struct {
	ImageName       string
	CacheDir        string
	EnableGPU       *bool
	LogLevel        string
	EnableBaremetal *bool
}

// ExtractCache pulls and extracts the cache image using the provided options
func ExtractCache(opts Options) error {
	if opts.ImageName == "" {
		return fmt.Errorf("image name must be specified")
	}

	if _, err := config.Initialize(config.ConfDir); err != nil {
		return fmt.Errorf("failed to initialize config: %w", err)
	}

	if err := logformat.ConfigureLogging(opts.LogLevel); err != nil {
		return fmt.Errorf("error configuring logging: %v", err)
	}

	if opts.EnableGPU != nil {
		config.SetEnabledGPU(*opts.EnableGPU)
		if !*opts.EnableGPU {
			logging.Debug("GPU checks disabled via client options")
		}
	}

	if opts.EnableBaremetal != nil {
		config.SetEnabledBaremetal(*opts.EnableBaremetal)
		if !*opts.EnableBaremetal {
			logging.Debug("Baremetal checks disabled via client options")
		}
	}

	cacheDir := opts.CacheDir
	if cacheDir == "" {
		cacheDir = constants.TritonCacheDir // default from init()
	}
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return fmt.Errorf("failed to create cache dir: %w", err)
	}
	constants.TritonCacheDir = cacheDir

	logging.Infof("Using Triton cache directory: %s", constants.TritonCacheDir)

	return fetcher.New().FetchAndExtractCache(opts.ImageName)
}

// GetSystemGPUInfo initializes the GPU accelerator and returns the list of available GPU devices
func GetSystemGPUInfo() ([]devices.TritonGPUInfo, error) {
	if _, err := config.Initialize(config.ConfDir); err != nil {
		return nil, fmt.Errorf("failed to initialize config: %w", err)
	}

	// Assume GPU is enabled if we're requesting GPU info
	if !config.IsGPUEnabled() {
		logging.Debug("Forcing ENABLE_GPU=true to retrieve GPU info")
		config.SetEnabledGPU(true)
	}

	// Initialize the GPU accelerator
	acc, err := accelerator.New(config.GPU, true)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize GPU accelerator: %w", err)
	}

	// Register the accelerator
	accelerator.GetRegistry().MustRegister(acc)

	// Fetch GPU device information
	info, err := preflightcheck.GetAllGPUInfo(acc)
	if err != nil {
		return nil, fmt.Errorf("failed to get GPU info: %w", err)
	}

	return info, nil
}
