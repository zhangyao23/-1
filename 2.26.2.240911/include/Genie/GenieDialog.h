//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief  Dialog API providing query functionality.
 */

#ifndef GENIE_DIALOG_H
#define GENIE_DIALOG_H

#include "GenieCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief A handle for dialog configuration instances.
 */
typedef const struct _GenieDialogConfig_Handle_t* GenieDialogConfig_Handle_t;

/**
 * @brief A handle for dialog instances.
 */
typedef const struct _GenieDialog_Handle_t* GenieDialog_Handle_t;

/**
 * @brief An enum which defines the sentence code supported by GENIE backends.
 */
typedef enum {
  /// The string is the entire query/response.
  GENIE_DIALOG_SENTENCE_COMPLETE = 0,
  /// The string is the beginning of the query/response.
  GENIE_DIALOG_SENTENCE_BEGIN = 1,
  /// The string is a part of the query/response and not the beginning or end.
  GENIE_DIALOG_SENTENCE_CONTINUE = 2,
  /// The string is the end of the query/response.
  GENIE_DIALOG_SENTENCE_END = 3,
  /// The query has been aborted.
  GENIE_DIALOG_SENTENCE_ABORT = 4,
} GenieDialog_SentenceCode_t;

/**
 * @brief A client-defined callback function to handle query responses.
 *
 * @param[in] response The query response.
 *
 * @param[in] sentenceCode The sentence code related to the responseStr.
 *
 * @param[in] userData The userData field provided to GenieDialog_query.
 *
 * @return None
 *
 */
typedef void (*GenieDialog_QueryCallback_t)(const char* response,
                                            const GenieDialog_SentenceCode_t sentenceCode,
                                            const void* userData);

//=============================================================================
// Functions
//=============================================================================

/**
 * @brief A function to create a dialog configuration from a JSON string.
 *
 * @param[in] str A configuration string. Must not be NULL.
 *
 * @param[out] configHandle A handle to the created config. Must not be NULL.
 *
 * @return Status code:
 *         - GENIE_STATUS_SUCCESS: API call was successful.
 *         - GENIE_STATUS_ERROR_INVALID_ARGUMENT: At least one argument is invalid.
 *         - GENIE_STATUS_ERROR_MEM_ALLOC: Memory allocation failure.
 *         - GENIE_STATUS_ERROR_INVALID_CONFIG: At least one configuration option is invalid.
 */
GENIE_API
Genie_Status_t GenieDialogConfig_createFromJson(const char* str,
                                                GenieDialogConfig_Handle_t* configHandle);

/**
 * @brief A function to free a dialog config.
 *
 * @param[in] configHandle A config handle.
 *
 * @return Status code:
 *         - GENIE_STATUS_SUCCESS: API call was successful.
 *         - GENIE_STATUS_ERROR_INVALID_HANDLE: Dialog handle is invalid.
 *         - GENIE_STATUS_ERROR_MEM_ALLOC: Memory (de)allocation failure.
 */
GENIE_API
Genie_Status_t GenieDialogConfig_free(const GenieDialogConfig_Handle_t configHandle);

/**
 * @brief A function to create a dialog. The dialog can be configured using a
 *        builder pattern.
 *
 * @param[in] configHandle A handle to a valid config. Must not be NULL.
 *
 * @param[out] dialogHandle A handle to the created dialog. Must not be NULL.
 *
 * @return Status code:
 *         - GENIE_STATUS_SUCCESS: API call was successful.
 *         - GENIE_STATUS_ERROR_INVALID_HANDLE: Config handle is invalid.
 *         - GENIE_STATUS_ERROR_INVALID_ARGUMENT: At least one argument is invalid.
 *         - GENIE_STATUS_ERROR_MEM_ALLOC: Memory allocation failure.
 */
GENIE_API
Genie_Status_t GenieDialog_create(const GenieDialogConfig_Handle_t configHandle,
                                  GenieDialog_Handle_t* dialogHandle);

/**
 * @brief A function to execute a query.
 *
 * @param[in] dialogHandle A dialog handle.
 *
 * @param[in] queryStr The input query.
 *
 * @param[in] sentenceCode The sentence code indicating the contents of the queryStr.
 *
 * @param[in] callback Callback function to handle query responses. Cannot be NULL.
 *
 * @param[in] userData User defined field provided in the query responses. Can be NULL.
 *
 * @return Status code:
 *         - GENIE_STATUS_SUCCESS: API call was successful.
 *         - GENIE_STATUS_ERROR_INVALID_HANDLE: Dialog handle is invalid.
 *         - GENIE_STATUS_ERROR_INVALID_ARGUMENT: At least one argument is invalid.
 *         - GENIE_STATUS_ERROR_MEM_ALLOC: Memory allocation failure.
 */
GENIE_API
Genie_Status_t GenieDialog_query(const GenieDialog_Handle_t dialogHandle,
                                 const char* queryStr,
                                 const GenieDialog_SentenceCode_t sentenceCode,
                                 const GenieDialog_QueryCallback_t callback,
                                 const void* userData);

/**
 * @brief A function to compute Perplexity from tokens.
 *
 * @param[in] dialogHandle A dialog handle.
 *
 * @param[in] queryTokens Pointer of tokens.
 *
 * @param[in] queryTokensSize Total number of tokens.
 *
 * @param[in] ppl API will update this to final Perplexity value.
 *
 * @return Status code:
 *         - GENIE_STATUS_SUCCESS: API call was successful.
 *         - GENIE_STATUS_ERROR_GENERAL: Memory allocation failure.
 */
GENIE_API
Genie_Status_t GenieDialog_PerplexityFromTokens(const GenieDialog_Handle_t dialogHandle,
                                                const int32_t* queryTokens,
                                                uint32_t queryTokensSize,
                                                float* ppl);

/**
 * @brief A function to compute Perplexity from text.
 *
 * @param[in] dialogHandle A dialog handle.
 *
 * @param[in] queryText Pointer of query text on Perplexity should compute.
 *
 * @param[in] ppl API will update this to final Perplexity value.
 *
 * @return Status code:
 *         - GENIE_STATUS_SUCCESS: API call was successful.
 *         - GENIE_STATUS_ERROR_GENERAL: Memory allocation failure.
 */
GENIE_API
Genie_Status_t GenieDialog_PerplexityFromText(const GenieDialog_Handle_t dialogHandle,
                                              const char* queryText,
                                              float* ppl);


/**
 * @brief A function to reset a dialog.
 *
 * @param[in] dialogHandle A dialog handle.
 *
 * @return Status code:
 *         - GENIE_STATUS_SUCCESS: API call was successful.
 *         - GENIE_STATUS_ERROR_INVALID_HANDLE: Dialog handle is invalid.
 */
GENIE_API
Genie_Status_t GenieDialog_reset(const GenieDialog_Handle_t dialogHandle);


/**
 * @brief A function to free a dialog.
 *
 * @param[in] dialogHandle A dialog handle.
 *
 * @return Status code:
 *         - GENIE_STATUS_SUCCESS: API call was successful.
 *         - GENIE_STATUS_ERROR_INVALID_HANDLE: Dialog handle is invalid.
 *         - GENIE_STATUS_ERROR_MEM_ALLOC: Memory (de)allocation failure.
 */
GENIE_API
Genie_Status_t GenieDialog_free(const GenieDialog_Handle_t dialogHandle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GENIE_DIALOG_H
