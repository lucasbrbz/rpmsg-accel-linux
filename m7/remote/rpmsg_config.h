/*
 * Copyright 2024 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef RPMSG_CONFIG_H_
#define RPMSG_CONFIG_H_

#define RL_MS_PER_INTERVAL         (1)
#define RL_BUFFER_PAYLOAD_SIZE     (496U)
#define RL_BUFFER_COUNT            (256U)
#define RL_API_HAS_ZEROCOPY        (1)
#define RL_USE_STATIC_API          (0)
#define RL_CLEAR_USED_BUFFERS      (0)
#define RL_USE_MCMGR_IPC_ISR_HANDLER (0)
#define RL_USE_ENVIRONMENT_CONTEXT (0)
#define RL_DEBUG_CHECK_BUFFERS     (0)

#endif /* RPMSG_CONFIG_H_ */
