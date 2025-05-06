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
package imgbuild

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/archive"
	"github.com/redhat-et/TKDK/cargohold/pkg/preflightcheck"
	"github.com/redhat-et/TKDK/cargohold/pkg/utils"
	logging "github.com/sirupsen/logrus"
)

type dockerBuilder struct{}

// Docker implementation of the ImageBuilder interface.
func (d *dockerBuilder) CreateImage(imageName, cacheDir string) error {
	wd, _ := os.Getwd()
	dockerfilePath := fmt.Sprintf("%s/Dockerfile", wd)
	tmpCacheDir := fmt.Sprintf("%s/io.triton.cache", wd)
	var allMetadata []CacheMetadataWithDummy

	// Copy cache contents into a directory within build context
	if err := os.MkdirAll(tmpCacheDir, 0755); err != nil {
		return fmt.Errorf("failed to create temp cache dir: %w", err)
	}
	defer os.RemoveAll(tmpCacheDir)

	err := copyDir(cacheDir+"/.", tmpCacheDir)
	if err != nil {
		return fmt.Errorf("failed to copy cacheDir into build context: %w", err)
	}

	jsonFiles, err := preflightcheck.FindAllTritonCacheJSON(tmpCacheDir)
	if err != nil {
		return fmt.Errorf("failed to find cache files: %w", err)
	}

	for _, jsonFile := range jsonFiles {
		data, ret := preflightcheck.GetTritonCacheJSONData(jsonFile)
		if ret != nil {
			return fmt.Errorf("failed to extract data from %s: %w", jsonFile, ret)
		}
		if data == nil {
			continue
		}

		dummyKey, ret := preflightcheck.ComputeDummyTritonKey(data)
		if ret != nil {
			return fmt.Errorf("failed to calculate dummy triton key for %s: %w", jsonFile, ret)
		}

		allMetadata = append(allMetadata, CacheMetadataWithDummy{
			Hash:       data.Hash,
			Backend:    data.Target.Backend,
			Arch:       preflightcheck.ConvertArchToString(data.Target.Arch),
			WarpSize:   data.Target.WarpSize,
			PTXVersion: data.PtxVersion,
			NumStages:  data.NumStages,
			NumWarps:   data.NumWarps,
			Debug:      data.Debug,
			DummyKey:   dummyKey,
		})
	}

	filepath.Walk(tmpCacheDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasPrefix(info.Name(), "__grp__") && strings.HasSuffix(info.Name(), ".json") {
			if err := utils.SanitizeGroupJSON(path); err != nil {
				logging.Warnf("could not sanitize %s: %v", path, err)
			}
		}
		return nil
	})

	err = generateDockerfile(imageName, tmpCacheDir, dockerfilePath)
	if err != nil {
		return fmt.Errorf("failed to generate Dockerfile: %w", err)
	}
	defer os.Remove(dockerfilePath)

	if _, err = os.Stat(dockerfilePath); os.IsNotExist(err) {
		return fmt.Errorf("dockerfile not found at %s", dockerfilePath)
	}

	apiClient, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return fmt.Errorf("failed to create Docker client: %w", err)
	}

	tar, err := archive.TarWithOptions(wd, &archive.TarOptions{IncludeSourceDir: false})
	if err != nil {
		return fmt.Errorf("error creating tar: %w", err)
	}
	defer tar.Close()

	metadataJSON, err := json.Marshal(allMetadata)
	if err != nil {
		return fmt.Errorf("failed to marshal cache metadata: %w", err)
	}

	labels := map[string]string{
		"cache.triton.image/metadata":    string(metadataJSON),
		"cache.triton.image/entry-count": strconv.Itoa(len(allMetadata)),
		"cache.triton.image/variant":     "multi",
	}
	buildOptions := types.ImageBuildOptions{
		Dockerfile: "Dockerfile",
		Tags:       []string{imageName},
		NoCache:    true,
		Remove:     false,
		Labels:     labels,
	}

	buildResponse, err := apiClient.ImageBuild(context.Background(), tar, buildOptions)
	if err != nil {
		return fmt.Errorf("error building image: %w", err)
	}
	defer buildResponse.Body.Close()

	_, err = io.Copy(os.Stdout, buildResponse.Body)
	if err != nil {
		return fmt.Errorf("error reading build output: %w", err)
	}

	imageWithTag := fmt.Sprintf("%s:%s", imageName, "latest")
	err = apiClient.ImageTag(context.Background(), imageName, imageWithTag)
	if err != nil {
		return fmt.Errorf("error tagging image: %w", err)
	}

	ret := utils.CleanupTmpDirs()
	if ret != nil {
		return fmt.Errorf("could not cleanup tmp dirs %v", ret)
	}
	logging.Info("Docker image built successfully")
	return nil
}
