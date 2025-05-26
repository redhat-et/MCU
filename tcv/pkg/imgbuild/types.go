package imgbuild

const DockerfileTemplate = `FROM scratch
LABEL org.opencontainers.image.title={{ .ImageTitle }}
COPY "{{ .CacheDir }}/." ./io.triton.cache/
COPY "{{ .ManifestDir }}/manifest.json" ./io.triton.manifest/manifest.json
`

type CacheMetadata struct {
	Hash       string `json:"hash"`
	Backend    string `json:"backend"`
	Arch       string `json:"arch"`
	WarpSize   int    `json:"warp_size"`
	PTXVersion *int   `json:"ptx_version,omitempty"`
	NumStages  int    `json:"num_stages,omitempty"`
	NumWarps   int    `json:"num_warps,omitempty"`
	Debug      bool   `json:"debug,omitempty"`
	DummyKey   string `json:"dummy_key"`
}

type DockerfileData struct {
	ImageTitle  string
	CacheDir    string
	ManifestDir string
}
