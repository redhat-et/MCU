/*
Copyright Istio Authors
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

package fetcher

import (
	"archive/tar"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/tarball"
	"github.com/google/go-containerregistry/pkg/v1/types"
	"github.com/hashicorp/go-multierror"
	logging "github.com/sirupsen/logrus"
	"github.com/tkdk/cargohold/pkg/accelerator"
	"github.com/tkdk/cargohold/pkg/config"
	"github.com/tkdk/cargohold/pkg/constants"
	"github.com/tkdk/cargohold/pkg/preflightcheck"
	"github.com/tkdk/cargohold/pkg/utils"
)

// A quick list of TODOS:
// 1. Add image caching to avoid the overhead of pulling the images down every time
// 2. Don't create directories/files in $HOME/.triton/cache if they already exist.

type tritonCacheExtractor struct {
	acc accelerator.Accelerator
}

type imgMgr struct {
	fetcher   ImgFetcher
	extractor TritonCacheExtractor
}

// TritonCacheExtractor extracts the Triton cache from an image.
type TritonCacheExtractor interface {
	ExtractCache(img v1.Image) error
}

// ImgMgr retrieves cache images.
type ImgMgr interface {
	FetchAndExtractCache(imgName string) error
}

// Factory function to create a new ImgMgr.
func New() ImgMgr {
	var a accelerator.Accelerator

	if config.IsGPUEnabled() {
		r := accelerator.GetRegistry()
		acc, err := accelerator.New(config.GPU, true)
		if err != nil {
			logging.Errorf("failed to init GPU accelerators: %v", err)
		} else {
			r.MustRegister(acc) // Register the accelerator with the registry
			a = acc
		}
		// defer accelerator.Shutdown() // TODO CALL IN CLEANUP
	}

	return &imgMgr{
		fetcher:   NewImgFetcher(),
		extractor: &tritonCacheExtractor{acc: a},
	}
}

type imgFetcher struct {
	fetcher Fetcher
}

type ImgFetcher interface {
	FetchImg(imgName string) (v1.Image, error)
}

// func saveImageLocally(path string, img v1.Image, ref name.Reference) error {
// 	out, err := os.Create(path)
// 	if err != nil {
// 		return fmt.Errorf("failed to create cache file: %w", err)
// 	}
// 	defer out.Close()

// 	err = tarball.WriteToFile(path, ref, img)
// 	if err != nil {
// 		return fmt.Errorf("failed to write image to cache: %w", err)
// 	}
// 	return nil
// }

func loadImageFromTarball(path string) (v1.Image, error) {
	img, err := tarball.ImageFromPath(path, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to load image from tarball: %w", err)
	}
	logging.Debug("loaded image from tarball!!!!!!!!")
	return img, nil
}

func NewImgFetcher() ImgFetcher {
	return &imgFetcher{fetcher: NewFetcher()}
}

// FetchImg pulls the image from the registry and extracts the TritonCache
func (i *imgFetcher) FetchImg(imgName string) (v1.Image, error) {
	if i.fetcher == nil {
		logging.Error("Error with fetcher!!!!!!!!")
		return nil, fmt.Errorf("failed to configure fetcher")
	}

	img, err := i.fetcher.FetchImg(imgName)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch image: %w", err)
	}
	logging.Debug("Img retrieved successfully!!!!!!!!")

	digest, err := img.Digest()
	if err != nil {
		return nil, fmt.Errorf("failed to get image digest: %w", err)
	}
	logging.Debugf("Img Digest: %s", digest)

	size, err := img.Size()
	if err != nil {
		return nil, fmt.Errorf("failed to get image digest: %w", err)
	}
	logging.Debugf("Img Size: %v\n", size)

	return img, nil
}

func (e *tritonCacheExtractor) ExtractCache(img v1.Image) error {
	// Handle Docker, OCI, and custom formats here.
	manifest, err := img.Manifest()
	if err != nil {
		return fmt.Errorf("failed to fetch manifest: %w", err)
	}

	if ret := preflightcheck.CompareTritonCacheImageToGPU(img, e.acc); ret != nil {
		return fmt.Errorf("***** the gpu and triton cache are incompatible ****")
	}

	if manifest.MediaType == types.DockerManifestSchema2 {
		// This case, assume we have docker images with "application/vnd.docker.distribution.manifest.v2+json"
		// as the manifest media type. Note that the media type of manifest is Docker specific and
		// all OCI images would have an empty string in .MediaType field.

		ret := extractDockerImg(img)
		if ret != nil {
			return fmt.Errorf("could not extract the Triton Cache from the container image %v", err)
		}

		ret = utils.CleanupTmpDirs()
		if ret != nil {
			return fmt.Errorf("could not cleanup tmp dirs %v", ret)
		}

		return nil
	}

	// We try to parse it as the "compat" variant image with a single "application/vnd.oci.image.layer.v1.tar+gzip" layer.
	errCompat := extractOCIStandardImg(img)
	if errCompat == nil {
		ret := utils.CleanupTmpDirs()
		if ret != nil {
			return fmt.Errorf("could not cleanup tmp dirs %v", ret)
		}
		return nil
	}

	// Otherwise, we try to parse it as the *oci* variant image with custom artifact media types.
	errOCI := extractOCIArtifactImg(img)
	if errOCI == nil {
		ret := utils.CleanupTmpDirs()
		if ret != nil {
			return fmt.Errorf("could not cleanup tmp dirs %v", ret)
		}
		return nil
	}

	ret := utils.CleanupTmpDirs()
	if ret != nil {
		return fmt.Errorf("could not cleanup tmp dirs %v", ret)
	}

	// We failed to parse the image in any format, so wrap the errors and return.
	return fmt.Errorf("the given image is in invalid format as an OCI image: %v",
		multierror.Append(err,
			fmt.Errorf("could not parse as compat variant: %v", errCompat),
			fmt.Errorf("could not parse as oci variant: %v", errOCI),
		),
	)
}

func (i *imgMgr) FetchAndExtractCache(imgName string) error {
	img, err := i.fetcher.FetchImg(imgName)
	if err != nil {
		return err
	}

	err = i.extractor.ExtractCache(img)
	if err != nil {
		return err
	}

	return nil
}

// extractOCIArtifactImg extracts the triton cache from the
// *oci* variant Triton Kernel Cache image:  //TODO ADD URL
func extractOCIArtifactImg(img v1.Image) error {
	layers, err := img.Layers()
	if err != nil {
		return fmt.Errorf("could not fetch layers: %v", err)
	}

	// The image must be single-layered.
	if len(layers) != 1 {
		return fmt.Errorf("number of layers must be 1 but got %d", len(layers))
	}

	// The layer type of the Triton cache itself in *oci* variant.
	const cacheLayerMediaType = "application/cache.triton.content.layer.v1+triton"

	// Find the target layer walking through the layers.
	var layer v1.Layer
	for _, l := range layers {
		mt, ret := l.MediaType()
		if ret != nil {
			return fmt.Errorf("could not retrieve the media type: %v", ret)
		}
		if mt == cacheLayerMediaType {
			layer = l
			break
		}
	}

	if layer == nil {
		return fmt.Errorf("could not find the layer of type %s", cacheLayerMediaType)
	}

	// Somehow go-container registry recognizes custom artifact layers as compressed ones,
	// while the GPU Kernel Cache/Binary layer is actually uncompressed and therefore
	// the content itself is a GPU Kernel Cache/Binary. So using "Uncompressed()" here result in errors
	// since internally it tries to umcompress it as gzipped blob.
	r, err := layer.Compressed()
	if err != nil {
		return fmt.Errorf("could not get layer content: %v", err)
	}
	defer r.Close()

	err = extractTritonCacheDirectory(r)
	if err != nil {
		return fmt.Errorf("could not extract Triton Kernel Cache: %v", err)
	}
	return nil
}

// extractDockerImg extracts the Triton Kernel Cache from the
// *compat* variant GPU Kernel Cache/Binary image with the standard Docker
// media type: application/vnd.docker.image.rootfs.diff.tar.gzip.
// https://github.com/maryamtahhan/cargohold/blob/main/spec-compat.md
func extractDockerImg(img v1.Image) error {
	layers, err := img.Layers()
	if err != nil {
		return fmt.Errorf("could not fetch layers: %v", err)
	}

	// The image must have at least one layer.
	if len(layers) == 0 {
		return errors.New("number of layers must be greater than zero")
	}

	layer := layers[len(layers)-1]
	mt, err := layer.MediaType()
	if err != nil {
		return fmt.Errorf("could not get media type: %v", err)
	}

	// Media type must be application/vnd.docker.image.rootfs.diff.tar.gzip.
	if mt != types.DockerLayer {
		return fmt.Errorf("invalid media type %s (expect %s)", mt, types.DockerLayer)
	}

	r, err := layer.Compressed()
	if err != nil {
		return fmt.Errorf("could not get layer content: %v", err)
	}
	defer r.Close()

	err = extractTritonCacheDirectory(r)
	if err != nil {
		return fmt.Errorf("could not extract Triton Kernel Cache: %v", err)
	}
	return nil
}

// extractOCIStandardImg extracts the Triton Kernel Cache from the
// *compat* variant Triton Kernel image with the standard OCI media type: application/vnd.oci.image.layer.v1.tar+gzip.
// https://github.com/maryamtahhan/cargohold/blob/main/spec-compat.md
func extractOCIStandardImg(img v1.Image) error {
	layers, err := img.Layers()
	if err != nil {
		return fmt.Errorf("could not fetch layers: %v", err)
	}

	// The image must have at least one layer.
	if len(layers) == 0 {
		return fmt.Errorf("number of layers must be greater than zero")
	}

	layer := layers[len(layers)-1]
	mt, err := layer.MediaType()
	if err != nil {
		return fmt.Errorf("could not get media type: %v", err)
	}

	// Check if the layer is "application/vnd.oci.image.layer.v1.tar+gzip".
	if types.OCILayer != mt {
		return fmt.Errorf("invalid media type %s (expect %s)", mt, types.OCILayer)
	}

	r, err := layer.Compressed()
	if err != nil {
		return fmt.Errorf("could not get layer content: %v", err)
	}
	defer r.Close()

	err = extractTritonCacheDirectory(r)
	if err != nil {
		return fmt.Errorf("could not extract Triton Kernel Cache: %v", err)
	}
	return nil
}

// Extracts the triton named "io.triton.cache" in a given reader for tar.gz.
// This is only used for *compat* variant.
// TODO add preflight checks here.
func extractTritonCacheDirectory(r io.Reader) error {
	gr, err := gzip.NewReader(r)
	if err != nil {
		return fmt.Errorf("failed to parse layer as tar.gz: %v", err)
	}

	tr := tar.NewReader(gr)
	// var cacheDirs []string  TODO RE-ENABLE

	for {
		h, err := tr.Next()
		if err == io.EOF {
			break // End of archive
		} else if err != nil {
			return fmt.Errorf("error reading tar archive: %w", err)
		}

		// Skip files not in the Triton cache directory
		if !strings.HasPrefix(h.Name, constants.TritonCacheDirName) {
			continue
		}

		// Track Triton cache directories
		relativePath := strings.TrimPrefix(h.Name, constants.TritonCacheDirName)
		if relativePath == "" {
			// cacheDirs = append(cacheDirs, h.Name) // Store the directory name TODO RE-ENABLE
			continue
		}

		filePath := filepath.Join(constants.TritonCacheDir, relativePath)

		switch h.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(filePath, os.FileMode(h.Mode)); err != nil {
				return fmt.Errorf("failed to create directory %s: %w", filePath, err)
			}
			// cacheDirs = append(cacheDirs, filePath) // Store created directory TODO RE-ENABLE

		case tar.TypeReg:
			if err := writeFile(filePath, tr, os.FileMode(h.Mode)); err != nil {
				return fmt.Errorf("failed to create file %s: %w", filePath, err)
			}

		default:
			logging.Debugf("Skipping unsupported type: %c in file %s\n", h.Typeflag, h.Name)
		}
	}

	return nil
}

// writeFile writes a file's content to disk from the tar reader
func writeFile(filePath string, tarReader io.Reader, mode os.FileMode) error {
	// Create any parent directories if needed
	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		return fmt.Errorf("failed to create parent directories for %s: %w", filePath, err)
	}

	outFile, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", filePath, err)
	}
	defer outFile.Close()

	if _, err := io.Copy(outFile, tarReader); err != nil {
		return fmt.Errorf("failed to copy content to file %s: %w", filePath, err)
	}

	if err := os.Chmod(filePath, mode); err != nil {
		return fmt.Errorf("failed to set file permissions for %s: %w", filePath, err)
	}

	return nil
}
