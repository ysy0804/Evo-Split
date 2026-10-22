#ifndef EVOSPLIT_RUNTIME_METRICS_H
#define EVOSPLIT_RUNTIME_METRICS_H

#include <cstdint>
#include <string>

// Device-side telemetry consumed by the online controller. The structure is
// intentionally runtime-neutral so an application can populate it from
// Android, Linux, or an existing monitoring agent.
struct PerformanceMetrics {
    uint64_t timestamp{0};

    double firstTokenLatencyMs{0.0};
    double avgInferenceLatencyMs{0.0};
    double p95LatencyMs{0.0};
    double fps{0.0};

    double memoryUsageMB{0.0};
    double memoryPeakMB{0.0};
    double pssMemoryMB{0.0};

    double cpuUsagePercent{0.0};
    double cpuFrequencyMHz{0.0};
    double gpuUsagePercent{0.0};
    double gpuFrequencyMHz{0.0};

    double powerConsumptionMw{0.0};
    double batteryCurrentMa{0.0};
    double batteryTempCelsius{0.0};
    double socTempCelsius{0.0};

    std::string backendInfo;
};

#endif
