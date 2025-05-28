package imgbuild

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/containers/buildah"
	"github.com/containers/common/pkg/config"
	is "github.com/containers/image/v5/storage"
	"github.com/containers/storage"
	"github.com/redhat-et/TKDK/tcv/pkg/constants"
	"github.com/redhat-et/TKDK/tcv/pkg/preflightcheck"
	"github.com/redhat-et/TKDK/tcv/pkg/utils"
	logging "github.com/sirupsen/logrus"
)

type buildahBuilder struct{}

func (b *buildahBuilder) CreateImage(imageName, cacheDir string) error {
	var allMetadata []CacheMetadata

	tmpCacheDir, err := os.MkdirTemp("", constants.BuildahCacheDirPrefix)
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpCacheDir)

	tmpManifestDir, err := os.MkdirTemp("", constants.BuildahManifestDirPrefix)
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpManifestDir)

	jsonFiles, err := preflightcheck.FindAllTritonCacheJSON(cacheDir)
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

		allMetadata = append(allMetadata, CacheMetadata{
			Hash:       data.Hash,
			Backend:    data.Target.Backend,
			Arch:       preflightcheck.ConvertArchToString(data.Target.Arch),
			WarpSize:   data.Target.WarpSize,
			PTXVersion: data.PtxVersion,
			DummyKey:   dummyKey,
			NumStages:  data.NumStages,
			NumWarps:   data.NumWarps,
			Debug:      data.Debug,
		})
	}

	manifestPath := filepath.Join(tmpManifestDir, "manifest.json")
	err = writeCacheManifest(manifestPath, allMetadata)
	if err != nil {
		return fmt.Errorf("failed to write manifest: %w", err)
	}

	err = copyDir(cacheDir, tmpCacheDir)
	if err != nil {
		return fmt.Errorf("error copying contents using cp: %v", err)
	}
	logging.Debugf("%s", tmpCacheDir)

	totalSize, err := getTotalDirSize(tmpCacheDir)
	if err != nil {
		return fmt.Errorf("failed to compute total cache size: %w", err)
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

	buildStoreOptions, _ := storage.DefaultStoreOptions()
	conf, err := config.Default()
	if err != nil {
		return fmt.Errorf("error configuring buildah: %v", err)
	}

	capabilitiesForRoot, err := conf.Capabilities("root", nil, nil)
	if err != nil {
		return fmt.Errorf("capabilitiesForRoot error: %v", err)
	}

	buildStore, err := storage.GetStore(buildStoreOptions)
	if err != nil {
		return fmt.Errorf("failed to init storage: %v", err)
	}

	defer func() {
		if _, err = buildStore.Shutdown(false); err != nil {
			logging.Errorf("shutdown failed: %v", err)
		}
	}()

	imageWithTag := imageName
	if !strings.Contains(imageName, ":") {
		imageWithTag = fmt.Sprintf("%s:latest", imageName)
	}

	imageRef, err := is.Transport.ParseStoreReference(buildStore, imageWithTag)
	if err != nil {
		return fmt.Errorf("error creating the image reference: %v", err)
	}

	builderOpts := buildah.BuilderOptions{
		Capabilities: capabilitiesForRoot,
		FromImage:    "scratch",
	}

	ctx := context.TODO()
	// Initialize Buildah
	builder, err := buildah.NewBuilder(ctx, buildStore, builderOpts)
	if err != nil {
		return fmt.Errorf("error creating Buildah builder: %v", err)
	}

	defer func() {
		if err = builder.Delete(); err != nil {
			logging.Errorf(" builder.Delete failed: %v", err)
		}
	}()

	summary, err := BuildTritonSummary(allMetadata)
	if err != nil {
		return fmt.Errorf("failed to build image summary: %w", err)
	}

	summaryJSON, err := json.Marshal(summary)
	if err != nil {
		return fmt.Errorf("failed to marshal summary for label: %w", err)
	}

	builder.SetLabel("cache.triton.image/variant", "multi")
	builder.SetLabel("cache.triton.image/entry-count", strconv.Itoa(len(allMetadata)))
	builder.SetLabel("cache.triton.image/summary", string(summaryJSON))
	builder.SetLabel("cache.triton.image/cache-size-bytes", strconv.FormatInt(totalSize, 10))
	addOptions := buildah.AddAndCopyOptions{}

	err = builder.Add("./io.triton.manifest/", false, addOptions, tmpManifestDir+"/.")
	if err != nil {
		return fmt.Errorf("error adding manifest %s to builder: %v", tmpManifestDir, err)
	}

	err = builder.Add("./io.triton.cache/", false, addOptions, tmpCacheDir+"/.")
	if err != nil {
		return fmt.Errorf("error adding %s to builder: %v", cacheDir, err)
	}

	commitOptions := buildah.CommitOptions{
		Squash: true,
	}

	imageID, _, _, err := builder.Commit(ctx, imageRef, commitOptions)
	if err != nil {
		return fmt.Errorf("error committing the image: %v", err)
	}

	logging.Infof("Image built! %s\n", imageID)
	ret := utils.CleanupTCVDirs()
	if ret != nil {
		return fmt.Errorf("could not cleanup tmp dirs %v", ret)
	}
	return nil
}

func copyDir(srcDir, dstDir string) error {
	cmd := exec.Command("cp", "-r", srcDir+"/.", dstDir)

	err := cmd.Run()
	if err != nil {
		return fmt.Errorf("error executing cp command: %v", err)
	}

	return nil
}
