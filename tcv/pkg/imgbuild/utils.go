package imgbuild

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/template"

	"github.com/redhat-et/TKDK/tcv/pkg/preflightcheck"
	logging "github.com/sirupsen/logrus"
)

func generateDockerfile(imageName, cacheDir, manifestDir, outputPath string) error {
	parts := strings.Split(imageName, "/")
	fullImageName := parts[len(parts)-1]
	imageTitle := strings.Split(fullImageName, ":")[0]

	data := DockerfileData{
		ImageTitle:  imageTitle,
		CacheDir:    cacheDir,
		ManifestDir: manifestDir,
	}

	tmpl, err := template.New("dockerfile").Parse(DockerfileTemplate)
	if err != nil {
		return fmt.Errorf("error parsing template: %w", err)
	}

	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("error creating Dockerfile: %w", err)
	}
	defer file.Close()

	if err := tmpl.Execute(file, data); err != nil {
		return fmt.Errorf("error executing template: %w", err)
	}

	logging.Infof("Dockerfile generated successfully at %s", outputPath)
	return nil
}

func writeCacheManifest(filePath string, data []CacheMetadata) error {
	bytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal manifest: %w", err)
	}

	if err := os.WriteFile(filePath, bytes, 0644); err != nil {
		return fmt.Errorf("failed to write manifest to %s: %w", filePath, err)
	}

	logging.Infof("Wrote manifest to %s", filePath)
	return nil
}

// BuildTritonSummary deduplicates kernel targets and produces a compact summary for labeling.
func BuildTritonSummary(metadata []CacheMetadata) (*preflightcheck.TritonSummary, error) {
	if len(metadata) == 0 {
		return nil, fmt.Errorf("no metadata provided to summarize")
	}

	seen := make(map[string]preflightcheck.SummaryTargetInfo)

	for _, entry := range metadata {
		key := fmt.Sprintf("%s-%s-%d", entry.Backend, entry.Arch, entry.WarpSize)
		if _, exists := seen[key]; !exists {
			seen[key] = preflightcheck.SummaryTargetInfo{
				Backend:  entry.Backend,
				Arch:     entry.Arch,
				WarpSize: entry.WarpSize,
			}
		}
	}

	// Convert map to slice
	var targets []preflightcheck.SummaryTargetInfo
	for _, v := range seen {
		targets = append(targets, v)
	}

	return &preflightcheck.TritonSummary{
		Targets: targets,
	}, nil
}

func getTotalDirSize(dir string) (int64, error) {
	var total int64
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			info, err := d.Info()
			if err != nil {
				return err
			}
			total += info.Size()
		}
		return nil
	})
	return total, err
}
