/*
Copyright 2022.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package config

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	logging "github.com/sirupsen/logrus"
)

var (
	instance *Config
	once     sync.Once
)

// Configuration structs
type TCVConfig struct {
	TCVNamespace     string
	EnabledGPU       *bool
	KubeConfig       string
	EnabledBaremetal *bool
}

type Config struct {
	KernelVersion float32
	TCV           TCVConfig
}

func newConfig() (*Config, error) {
	// Ensure the directory exists or create it
	absBaseDir, err := filepath.Abs(ConfDir)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path for config-dir: %s: %w", ConfDir, err)
	}

	// Check if the directory exists
	s, err := os.Stat(absBaseDir)
	if os.IsNotExist(err) {
		// If it doesn't exist, create it
		if err = os.MkdirAll(absBaseDir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create config-dir %s: %w", ConfDir, err)
		}
		s, err = os.Stat(absBaseDir)
		if err != nil {
			return nil, fmt.Errorf("failed to stat config-dir %s: %w", ConfDir, err)
		}
	}
	if !s.IsDir() {
		return nil, fmt.Errorf("config-dir %s is not a directory", ConfDir)
	}

	// Proceed to create and return the Config instance
	return &Config{
		TCV:           getTCVConfig(),
		KernelVersion: float32(0),
	}, nil
}

// Instance returns the singleton Config instance
func Instance() *Config {
	return instance
}

// Initialize initializes the global instance once and returns an error if
func Initialize(baseDir string) (*Config, error) {
	var err error
	once.Do(func() {
		ConfDir = baseDir
		instance, err = newConfig()
	})
	return instance, err
}

func getTCVConfig() TCVConfig {
	var gpu, bm *bool

	// GPU: default to true unless explicitly set to "false"
	if val, exists := os.LookupEnv("ENABLE_GPU"); exists {
		b := strings.EqualFold(val, "true")
		gpu = &b
	} else {
		b := true // default to true
		gpu = &b
	}

	if val, exists := os.LookupEnv("ENABLE_BAREMETAL"); exists {
		b := strings.EqualFold(val, "true")
		bm = &b
	} else {
		b := false
		bm = &b
	}

	return TCVConfig{
		EnabledGPU:       gpu,
		EnabledBaremetal: bm,
		TCVNamespace:     getConfig("KEPLER_NAMESPACE", defaultNamespace),
		KubeConfig:       getConfig("KUBE_CONFIG", defaultKubeConfig),
	}
}

// getConfig returns the value of the key by first looking in the environment
// and then in the config file if it exists or else returns the default value.
func getConfig(key, defaultValue string) string {
	// env var takes precedence over config file
	if envValue, exists := os.LookupEnv(key); exists {
		return envValue
	}

	// return config file value if there is one
	configFile := filepath.Join(ConfDir, key)
	if value, err := os.ReadFile(configFile); err == nil {
		return strings.TrimSpace(bytes.NewBuffer(value).String())
	}

	return defaultValue
}

func logBoolConfigs() {
	logging.Infof("ENABLE_GPU: %t", IsGPUEnabled())
	logging.Infof("ENABLE_BAREMETAL: %t", IsBaremetalEnabled())
}

func LogConfigs() {
	logging.Infof("config-dir: %s", ConfDir)
	logBoolConfigs()
}

func SetEnabledGPU(enabled bool) {
	b := enabled
	instance.TCV.EnabledGPU = &b
}

func SetEnabledBaremetal(enabled bool) {
	b := enabled
	instance.TCV.EnabledBaremetal = &b
}

// SetKubeConfig set kubeconfig file
func SetKubeConfig(k string) {
	instance.TCV.KubeConfig = k
}

func KubeConfig() string {
	return instance.TCV.KubeConfig
}

func IsGPUEnabled() bool {
	if instance.TCV.EnabledGPU != nil {
		return *instance.TCV.EnabledGPU
	}
	return false
}

func IsBaremetalEnabled() bool {
	if instance.TCV.EnabledBaremetal != nil {
		return *instance.TCV.EnabledBaremetal
	}
	return false
}
