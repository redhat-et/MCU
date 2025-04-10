package build

var (
	// Version is the version of cargohold. Set by the linker flags in the Makefile.
	Version string
	// Revision is the Git commit that was compiled. Set by the linker flags in the Makefile.
	Revision string
	// Branch is the Git branch that was compiled. Set by the linker flags in the Makefile.
	Branch string
	// OS is the operating system cargohold was built for. Set by the linker flags in the Makefile.
	OS string
	// Arch is the architecture cargohold was built for. Set by the linker flags in the Makefile.
	Arch string
)
