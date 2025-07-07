//==============================================================================
//
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#ifndef BUILD_OPTIONS_H
#define BUILD_OPTIONS_H 1

namespace build_options {
#ifdef WITH_OPT_DEBUG
#ifndef DEFOPT_LOG
#define DEFOPT_LOG 1
#endif
constexpr bool WithDebugOpt = true;
#else
constexpr bool WithDebugOpt = false;
#endif

#ifdef DEFOPT_LOG
constexpr bool DefOptLog = true;
#else
constexpr bool DefOptLog = false;
#endif

#ifdef DEBUG_TILING
constexpr bool DebugTiling = true;
#else
constexpr bool DebugTiling = false;
#endif

#ifdef PREPARE_DISABLED
static constexpr bool WITH_PREPARE = false;
#else
static constexpr bool WITH_PREPARE = true;
#endif

#ifdef DEBUG_REGISTRY
constexpr bool DebugRegistry = true;
#else
constexpr bool DebugRegistry = false;
#endif

#ifdef DEBUG_TCM
constexpr bool DebugTcm = true;
#else
constexpr bool DebugTcm = false;
#endif

#ifdef __hexagon__
constexpr bool IsPlatformHexagon = true;
#else
constexpr bool IsPlatformHexagon = false;
#endif

#ifdef _WIN32
constexpr bool PLATFORM_CANNOT_BYPASS_VCALL = true;
constexpr bool OS_WIN = true;
#else
constexpr bool PLATFORM_CANNOT_BYPASS_VCALL = false;
constexpr bool OS_WIN = false;
#endif

// For TID preemption, ResMgr implements preemption of worker threads and HMX, but the TIDs of all
// workers and the main thread must be explicitly specified to ResMgr.
#ifdef TID_PREEMPTION
constexpr bool TidPreemption = true;
#else
constexpr bool TidPreemption = false;
#endif

// Enable if we have proper preemption and enough threads (v81 or later).
//#define DUAL_GRAPH
#ifdef DUAL_GRAPH
constexpr bool DualGraph = true;
constexpr unsigned MaxGraphs = 2u;
#else
constexpr bool DualGraph = false;
constexpr unsigned MaxGraphs = 1u;
#endif

#if HEX_ARCH >= 79
constexpr bool HasUserPmus = true;
#else
constexpr bool HasUserPmus = false;
#endif

#if HEX_ARCH >= 85
#define HAS_ELTWISE 1
constexpr bool HasEltwise = true;
#else
constexpr bool HasEltwise = false;
#define HAS_ELTWISE 0
#endif

#if WITH_QNN_HTP_AUTO
constexpr bool EnableLazyLinking = false;
#else
constexpr bool EnableLazyLinking = true;

#endif

} // namespace build_options

#endif // BUILD_OPTIONS_H
