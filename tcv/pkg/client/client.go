package client

import (
	"fmt"
	"os"

	"github.com/redhat-et/TKDK/tcv/pkg/config"
	"github.com/redhat-et/TKDK/tcv/pkg/constants"
	"github.com/redhat-et/TKDK/tcv/pkg/fetcher"
	"github.com/redhat-et/TKDK/tcv/pkg/logformat"
	logging "github.com/sirupsen/logrus"
)

type Options struct {
	ImageName       string
	CacheDir        string
	EnableGPU       bool
	LogLevel        string
	EnableBaremetal bool
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

	config.SetEnabledGPU(opts.EnableGPU)
	if !opts.EnableGPU {
		logging.Debug("GPU checks disabled via client options")
	}

	config.SetEnabledBaremetal(opts.EnableBaremetal)
	if !opts.EnableBaremetal {
		logging.Debug("Baremetal checks disabled via client options")
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

func loggingLevelFromString(level string) error {
	if level == "" {
		level = "info"
	}
	lvl, err := logging.ParseLevel(level)
	if err != nil {
		return err
	}
	logging.SetLevel(lvl)
	logging.SetReportCaller(true)
	return nil
}
