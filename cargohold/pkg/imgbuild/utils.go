package imgbuild

import (
	"fmt"
	"os"
	"strings"
	"text/template"

	logging "github.com/sirupsen/logrus"
)

const DockerfileTemplate = `FROM scratch
LABEL org.opencontainers.image.title={{ .ImageTitle }}
COPY "{{ .CacheDir }}/." ./io.triton.cache/
`

type DockerfileData struct {
	ImageTitle string
	CacheDir   string
}

func generateDockerfile(imageName, cacheDir, outputPath string) error {
	parts := strings.Split(imageName, "/")
	fullImageName := parts[len(parts)-1]
	imageTitle := strings.Split(fullImageName, ":")[0]

	data := DockerfileData{
		ImageTitle: imageTitle,
		CacheDir:   cacheDir,
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
