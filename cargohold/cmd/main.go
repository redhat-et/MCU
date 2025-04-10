/*
Copyright Red Hat Inc.

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

package main

import (
	"fmt"
	"os"

	"github.com/containers/buildah"
	"github.com/containers/storage/pkg/unshare"
	logging "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/tkdk/cargohold/pkg/config"
	"github.com/tkdk/cargohold/pkg/fetcher"
	"github.com/tkdk/cargohold/pkg/imgbuild"
	"github.com/tkdk/cargohold/pkg/logformat"
	"github.com/tkdk/cargohold/pkg/utils"
)

const (
	exitNormal       = 0
	exitExtractError = 1
	exitCreateError  = 2
	exitLogError     = 3
)

func getCacheImage(imageName string) error {
	f := fetcher.New()
	return f.FetchAndExtractCache(imageName)
}

func createCacheImage(imageName, cacheDir string) error {
	_, err := utils.FilePathExists(cacheDir)
	if err != nil {
		return fmt.Errorf("error checking cache file path: %v", err)
	}

	builder, _ := imgbuild.New()
	if builder == nil {
		return fmt.Errorf("failed to create builder")
	}

	err = builder.CreateImage(imageName, cacheDir)
	if err != nil {
		return fmt.Errorf("failed to create the OCI image: %v", err)
	}

	logging.Info("OCI image created successfully.")
	return nil
}

func main() {
	var imageName string
	var cacheDirName string
	var createFlag bool
	var extractFlag bool
	var baremetalFlag bool
	var logLevel string

	logging.SetReportCaller(true)
	logging.SetFormatter(logformat.Default)

	// Initialize the config
	_, err := config.Initialize(config.ConfDir)
	if err != nil {
		logging.Fatalf("Error initializing config: %v\n", err)
		os.Exit(exitLogError)
	}

	var rootCmd = &cobra.Command{
		Use:   "cargohold",
		Short: "A GPU Kernel runtime container image management utility",
		PersistentPreRun: func(cmd *cobra.Command, args []string) {
			// logging
			if err := logformat.ConfigureLogging(logLevel); err != nil {
				logging.Errorf("Error configuring logging: %v", err)
				os.Exit(exitLogError)
			}
		},
		Run: func(cmd *cobra.Command, args []string) {
			config.SetEnabledBaremetal(baremetalFlag)
			logging.Infof("baremetalFlag %v", baremetalFlag)
			if createFlag {
				if err := createCacheImage(imageName, cacheDirName); err != nil {
					logging.Errorf("Error creating image: %v\n", err)
					os.Exit(exitCreateError)
				}
			}

			if extractFlag {
				if err := getCacheImage(imageName); err != nil {
					logging.Errorf("Error extracting image: %v\n", err)
					os.Exit(exitExtractError)
				}
			}

			if !createFlag && !extractFlag {
				logging.Error("No action specified. Use --create or --extract flag.")
				os.Exit(exitNormal)
			}
		},
	}

	// Define flags for Cobra
	rootCmd.Flags().BoolVarP(&baremetalFlag, "baremetal", "b", false, "Run baremetal preflight checks")
	rootCmd.Flags().StringVarP(&imageName, "image", "i", "", "OCI image name")
	rootCmd.Flags().StringVarP(&cacheDirName, "dir", "d", "", "Triton Cache Directory")
	rootCmd.Flags().BoolVarP(&createFlag, "create", "c", false, "Create OCI image")
	rootCmd.Flags().BoolVarP(&extractFlag, "extract", "e", false, "Extract a Triton cache from an OCI image")
	rootCmd.Flags().StringVarP(&logLevel, "log-level", "l", "", "Set the logging verbosity level: debug, info, warning or error")

	// Ensure the image flag is required
	ret := rootCmd.MarkFlagRequired("image")
	if ret != nil {
		logging.Fatalf("Error: %v\n", ret)
	}
	// Important to call from main()
	if buildah.InitReexec() {
		return
	}
	unshare.MaybeReexecUsingUserNamespace(false)

	config.SetEnabledGPU(true) // ASSUME TRUE FOR NOW

	// Execute the Cobra command
	if err := rootCmd.Execute(); err != nil {
		logging.Fatalf("Error: %v\n", err)
	}
}
