package devices

import (
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strconv"

	logging "github.com/sirupsen/logrus"
	"github.com/tkdk/cargohold/pkg/config"
	"github.com/tkdk/cargohold/pkg/utils"
)

const amdHwType = config.GPU

var (
	amdAccImpl = gpuAMD{}
	amdType    DeviceType
)

type gpuAMD struct {
	devices map[int]GPUDevice
}

type AMDGPUInfo struct {
	GPUInfo  map[int]*AMDCardInfo
	ListInfo map[int]*AMDListInfo
}

type AMDCardInfo struct {
	GPU              int            `json:"gpu"`
	ASIC             AMDASIC        `json:"asic"`
	Bus              AMDBus         `json:"bus"`
	VBIOS            AMDVBIOS       `json:"vbios"`
	Driver           AMDDriver      `json:"driver"`
	Board            AMDBoard       `json:"board"`
	RAS              AMDRAS         `json:"ras"`
	Partition        AMDPartition   `json:"partition"`
	SOCPState        string         `json:"soc_pstate"`
	XGMIPlpd         AMDXGMIPlpd    `json:"xgmi_plpd"`
	ProcessIsolation string         `json:"process_isolation"`
	NUMA             AMDNUMA        `json:"numa"`
	VRAM             AMDVRAM        `json:"vram"`
	CacheInfo        []AMDCacheInfo `json:"cache_info"`
}

type AMDASIC struct {
	MarketName            string `json:"market_name"`
	VendorID              string `json:"vendor_id"`
	VendorName            string `json:"vendor_name"`
	SubvendorID           string `json:"subvendor_id"`
	DeviceID              string `json:"device_id"`
	SubsystemID           string `json:"subsystem_id"`
	RevID                 string `json:"rev_id"`
	ASICSerial            string `json:"asic_serial"`
	OAMID                 string `json:"oam_id"`
	NumComputeUnits       int    `json:"num_compute_units"`
	TargetGraphicsVersion string `json:"target_graphics_version"`
}

type AMDBus struct {
	BDF                  string `json:"bdf"`
	MaxPCIeWidth         int    `json:"max_pcie_width"`
	PCIeInterfaceVersion string `json:"pcie_interface_version"`
	SlotType             string `json:"slot_type"`
}

type AMDVBIOS struct {
	Name       string `json:"name"`
	BuildDate  string `json:"build_date"`
	PartNumber string `json:"part_number"`
	Version    string `json:"version"`
}

type AMDDriver struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type AMDBoard struct {
	ModelNumber      string `json:"model_number"`
	ProductSerial    string `json:"product_serial"`
	FRUID            string `json:"fru_id"`
	ProductName      string `json:"product_name"`
	ManufacturerName string `json:"manufacturer_name"`
}

type AMDRAS struct {
	EEPROMVersion   string            `json:"eeprom_version"`
	ParitySchema    string            `json:"parity_schema"`
	SingleBitSchema string            `json:"single_bit_schema"`
	DoubleBitSchema string            `json:"double_bit_schema"`
	PoisonSchema    string            `json:"poison_schema"`
	ECCBlockState   map[string]string `json:"ecc_block_state"`
}

type AMDPartition struct {
	ComputePartition string `json:"compute_partition"`
	MemoryPartition  string `json:"memory_partition"`
	PartitionID      int    `json:"partition_id"`
}

type AMDXGMIPlpd struct {
	NumSupported int       `json:"num_supported"`
	CurrentID    int       `json:"current_id"`
	PLPDs        []AMDPLPD `json:"plpds"`
}

type AMDPLPD struct {
	PolicyID          int    `json:"policy_id"`
	PolicyDescription string `json:"policy_description"`
}

type AMDNUMA struct {
	Node     int `json:"node"`
	Affinity int `json:"affinity"`
}

type AMDVRAM struct {
	Type     string      `json:"type"`
	Vendor   string      `json:"vendor"`
	Size     AMDVRAMSize `json:"size"`
	BitWidth int         `json:"bit_width"`
}

type AMDVRAMSize struct {
	Value int    `json:"value"`
	Unit  string `json:"unit"`
}

type AMDCacheInfo struct {
	Cache            int          `json:"cache"`
	CacheProperties  []string     `json:"cache_properties"`
	CacheSize        AMDUnitValue `json:"cache_size"`
	CacheLevel       int          `json:"cache_level"`
	MaxNumCUShared   int          `json:"max_num_cu_shared"`
	NumCacheInstance int          `json:"num_cache_instance"`
}

type AMDUnitValue struct {
	Value float64 `json:"value"`
	Unit  string  `json:"unit"`
}

type AMDListInfo struct {
	GPU         int    `json:"gpu"`
	BDF         string `json:"bdf"`
	UniqueID    string `json:"uuid"`
	KFDID       int    `json:"kfd_id"`
	NodeID      int    `json:"node_id"`
	PartitionID int    `json:"partition_id"`
}

func amdCheck(r *Registry) {
	if err := initAMDLib(); err != nil {
		logging.Debugf("Error initializing AMD SMI: %v", err)
		return
	}
	amdType = AMD
	if err := addDeviceInterface(r, amdType, amdHwType, amdDeviceStartup); err == nil {
		logging.Infof("Using %s to obtain GPU info", amdAccImpl.Name())
	} else {
		logging.Infof("Error registering amd-smi: %v", err)
	}
}

func amdDeviceStartup() Device {
	a := amdAccImpl
	if err := a.InitLib(); err != nil {
		logging.Errorf("Error initializing %s: %v", amdType.String(), err)
		return nil
	}
	if err := a.Init(); err != nil {
		logging.Errorf("Failed to init device: %v", err)
		return nil
	}
	logging.Infof("Using %s to obtain GPU info", amdType.String())
	return &a
}

func initAMDLib() error {
	if utils.HasApp("amd-smi") {
		return nil
	}
	return errors.New("couldn't find amd-smi")
}

func (r *gpuAMD) InitLib() error {
	return initAMDLib()
}

func (r *gpuAMD) Name() string {
	return amdType.String()
}

func (r *gpuAMD) DevType() DeviceType {
	return amdType
}

func (r *gpuAMD) HwType() string {
	return amdHwType
}

func (r *gpuAMD) Init() error {
	gpuInfoList, err := getAllAMDGPUInfo()
	if err != nil {
		return fmt.Errorf("failed to get GPU information: %v", err)
	}

	r.devices = make(map[int]GPUDevice, len(gpuInfoList.GPUInfo))
	for gpuID, info := range gpuInfoList.GPUInfo {
		memTotal := calculateMemoryMB(info.VRAM.Size.Value, info.VRAM.Size.Unit)
		name := "card" + strconv.Itoa(gpuID)
		r.devices[gpuID] = GPUDevice{
			ID: gpuID,
			TritonInfo: TritonGPUInfo{
				Name:              name,
				UUID:              gpuInfoList.ListInfo[gpuID].UniqueID,
				ComputeCapability: "",
				Arch:              info.ASIC.TargetGraphicsVersion,
				WarpSize:          64,
				MemoryTotalMB:     memTotal,
				Backend:           "hip",
			},
		}
	}
	return nil
}

// Converts VRAM size to MB, handling different units
func calculateMemoryMB(value int, unit string) uint64 {
	switch unit {
	case "GB":
		return uint64(value * 1024)
	case "KB":
		return uint64(value / 1024)
	default: // Default to MB
		return uint64(value)
	}
}
func (r *gpuAMD) Shutdown() bool {
	return true
}

func getAllAMDGPUInfo() (*AMDGPUInfo, error) {
	gpus, err := getAMDGPUInfo()
	if err != nil {
		return nil, fmt.Errorf("could not get GPU info")
	}
	list, err := getAMDListInfo()
	if err != nil {
		return nil, fmt.Errorf("could not get system info")
	}
	return &AMDGPUInfo{
		GPUInfo:  gpus,
		ListInfo: list,
	}, nil
}

func getAMDGPUInfo() (map[int]*AMDCardInfo, error) {
	cmd := exec.Command("amd-smi", "static", "--json")
	output, err := cmd.Output()
	if err != nil {
		logging.Debugf("failed to execute amd-smi: %v", err)
		return nil, fmt.Errorf("failed to execute amd-smi: %v", err)
	}

	var gpuInfo []*AMDCardInfo
	if err := json.Unmarshal(output, &gpuInfo); err != nil {
		logging.Debugf("failed to parse amd-smi output: %v", err)
		return nil, fmt.Errorf("failed to parse amd-smi output: %v", err)
	}

	parsedGPUs := make(map[int]*AMDCardInfo)
	for _, gpu := range gpuInfo {
		parsedGPUs[gpu.GPU] = gpu
	}
	return parsedGPUs, nil
}

func getAMDListInfo() (map[int]*AMDListInfo, error) {
	cmd := exec.Command("amd-smi", "list", "--json")
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to execute amd-smi: %v", err)
	}

	var listInfo []*AMDListInfo
	if err = json.Unmarshal(output, &listInfo); err != nil {
		return nil, fmt.Errorf("failed to parse amd-smi output: %v", err)
	}

	parsedGPUs := make(map[int]*AMDListInfo)
	for _, gpu := range listInfo {
		parsedGPUs[gpu.GPU] = gpu
	}
	return parsedGPUs, nil
}

func (r *gpuAMD) GetAllGPUInfo() ([]TritonGPUInfo, error) {
	var allTritonInfo []TritonGPUInfo
	for gpuID, dev := range r.devices {
		allTritonInfo = append(allTritonInfo, dev.TritonInfo)
		logging.Debugf("GPU %d: %+v", gpuID, dev.TritonInfo)
	}
	return allTritonInfo, nil
}

func (r *gpuAMD) GetGPUInfo(gpuID int) (TritonGPUInfo, error) {
	dev, exists := r.devices[gpuID]
	if !exists {
		return TritonGPUInfo{}, fmt.Errorf("GPU device %d not found", gpuID)
	}
	return dev.TritonInfo, nil
}
