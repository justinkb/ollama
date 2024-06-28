package gpu

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"

	"github.com/ollama/ollama/format"
)

var (
	// Used to validate if the given ROCm lib is usable
	ROCmLibGlobs          = []string{"libhipblas.so.2*", "rocblas"} // TODO - probably include more coverage of files here...
	RocmStandardLocations = []string{"/opt/rocm/lib", "/usr/lib64"}
)

// Gather GPU information from the amdgpu driver if any supported GPUs are detected
func AMDGetGPUInfo() []RocmGPUInfo {
	resp := []RocmGPUInfo{}

	usedFile := "/mnt/c/Users/Paul/Documents/sys-class-drm-card1-device-mem_info_vram_used.txt"
	totalMemory := uint64(25753026560)
	usedMemory, err := getFreeMemory(usedFile)

	gpuInfo := RocmGPUInfo{
		GpuInfo: GpuInfo{
			Library: "rocm",
			memInfo: memInfo{
				TotalMemory: totalMemory,
				FreeMemory:  totalMemory-usedMemory,
			},
			ID:            "0",
			Name:          "1002:744c",
			Compute:       "gfx1100",
			MinimumMemory: rocmMinimumMemory,
			DriverMajor:   0,
			DriverMinor:   0,
		},
		usedFilepath: usedFile,
	}

	// Final validation is gfx compatibility - load the library if we haven't already loaded it
	// even if the user overrides, we still need to validate the library
	libDir, err := AMDValidateLibDir()
	if err != nil {
		slog.Warn("unable to verify rocm library, will use cpu", "error", err)
		return nil
	}
	gpuInfo.DependencyPath = libDir

	// The GPU has passed all the verification steps and is supported
	resp = append(resp, gpuInfo)

	if len(resp) == 0 {
		slog.Info("no compatible amdgpu devices detected")
	}
	return resp
}

// Prefer to use host installed ROCm, as long as it meets our minimum requirements
// failing that, tell the user how to download it on their own
func AMDValidateLibDir() (string, error) {
	libDir, err := commonAMDValidateLibDir()
	if err == nil {
		return libDir, nil
	}

	// Well known ollama installer path
	installedRocmDir := "/usr/share/ollama/lib/rocm"
	if rocmLibUsable(installedRocmDir) {
		return installedRocmDir, nil
	}

	// If we still haven't found a usable rocm, the user will have to install it on their own
	slog.Warn("amdgpu detected, but no compatible rocm library found.  Either install rocm v6, or follow manual install instructions at https://github.com/ollama/ollama/blob/main/docs/linux.md#manual-install")
	return "", fmt.Errorf("no suitable rocm found, falling back to CPU")
}

func (gpus RocmGPUInfoList) RefreshFreeMemory() error {
	if len(gpus) == 0 {
		return nil
	}
	for i := range gpus {
		usedMemory, err := getFreeMemory(gpus[i].usedFilepath)
		if err != nil {
			return err
		}
		slog.Debug("updating rocm free memory", "gpu", gpus[i].ID, "name", gpus[i].Name, "before", format.HumanBytes2(gpus[i].FreeMemory), "now", format.HumanBytes2(gpus[i].TotalMemory-usedMemory))
		gpus[i].FreeMemory = gpus[i].TotalMemory - usedMemory
	}
	return nil
}

func getFreeMemory(usedFile string) (uint64, error) {
	buf, err := os.ReadFile(usedFile)
	if err != nil {
		return 0, fmt.Errorf("failed to read sysfs node %s %w", usedFile, err)
	}
	usedMemory, err := strconv.ParseUint(strings.TrimSpace(string(buf)), 10, 64)
	if err != nil {
		slog.Debug("failed to parse sysfs node", "file", usedFile, "error", err)
		return 0, fmt.Errorf("failed to parse sysfs node %s %w", usedFile, err)
	}
	return usedMemory, nil
}
