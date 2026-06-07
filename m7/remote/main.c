/*
 * Copyright 2024 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "board.h"
#include "fsl_debug_console.h"
#include "FreeRTOS.h"
#include "task.h"
#include "rpmsg_lite.h"
#include "rpmsg_queue.h"
#include "rpmsg_ns.h"
#include "app.h"

/*******************************************************************************
 * Definitions
 ******************************************************************************/
#define APP_TASK_STACK_SIZE  (1024U)
#define LOCAL_EPT_ADDR       (30U)
#define ONE_G                16384
#define SAMPLE_DELAY_MS      100U
#define NORMAL_SAMPLES       40U
#define IMBALANCE_SAMPLES    40U
#define ANOMALY_SAMPLES      20U

typedef enum { STATE_NORMAL = 0, STATE_IMBALANCE, STATE_ANOMALY } accel_state_t;

typedef struct __attribute__((packed))
{
    int16_t x;
    int16_t y;
    int16_t z;
    uint8_t state;   /* 0=NORMAL 1=IMBALANCE 2=ANOMALY */
    uint8_t padding;
} accel_msg_t;

/*******************************************************************************
 * Simulation helpers
 ******************************************************************************/
static uint16_t s_lfsr = 0xACE1U;

static int16_t rand_noise(int16_t amplitude)
{
    s_lfsr ^= s_lfsr >> 7U;
    s_lfsr ^= (uint16_t)(s_lfsr << 9U);
    s_lfsr ^= s_lfsr >> 13U;
    return (int16_t)((int16_t)(s_lfsr % (uint16_t)(2 * amplitude + 1)) - amplitude);
}

static accel_msg_t make_sample(accel_state_t state)
{
    accel_msg_t msg = {0};
    msg.state       = (uint8_t)state;
    switch (state)
    {
        case STATE_NORMAL:
            msg.x = rand_noise(80);
            msg.y = rand_noise(80);
            msg.z = (int16_t)(ONE_G + rand_noise(80));
            break;
        case STATE_IMBALANCE:
            msg.x = (int16_t)(8192  + rand_noise(250));
            msg.y = (int16_t)(1500  + rand_noise(250));
            msg.z = (int16_t)(14189 + rand_noise(250));
            break;
        case STATE_ANOMALY:
            msg.x = (int16_t)(20000  + rand_noise(12000));
            msg.y = (int16_t)(-18000 + rand_noise(10000));
            msg.z = (int16_t)(5000   + rand_noise(15000));
            break;
    }
    return msg;
}

/*******************************************************************************
 * Main task
 ******************************************************************************/
static void accel_task(void *param)
{
    static const accel_state_t seq[]    = {STATE_NORMAL, STATE_IMBALANCE,
                                           STATE_NORMAL,  STATE_ANOMALY};
    static const uint32_t      limits[] = {NORMAL_SAMPLES, IMBALANCE_SAMPLES,
                                           NORMAL_SAMPLES, ANOMALY_SAMPLES};
    static const char         *names[]  = {"NORMAL", "IMBALANCE", "ANOMALY"};

    PRINTF("\r\nRPMsg accel simulator (M7 -> A53)\r\n");
    PRINTF("Shared mem base: 0x%x\r\n", RPMSG_LITE_SHMEM_BASE);

    PRINTF("Calling rpmsg_lite_remote_init...\r\n");
    struct rpmsg_lite_instance *rpmsg = rpmsg_lite_remote_init(
        (void *)RPMSG_LITE_SHMEM_BASE, RPMSG_LITE_LINK_ID, RL_NO_FLAGS);
    PRINTF("rpmsg_lite_remote_init returned: 0x%08X\r\n", (uint32_t)rpmsg);

    if (rpmsg == RL_NULL)
    {
        PRINTF("ERROR: rpmsg_lite_remote_init failed!\r\n");
        for (;;) {}
    }

    PRINTF("Waiting for Linux link up...\r\n");
    PRINTF("  link_state addr=0x%08X initial=%u\r\n",
           (uint32_t)&rpmsg->link_state,
           *(volatile uint32_t *)&rpmsg->link_state);
    {
        uint32_t poll = 0U;
        while (*(volatile uint32_t *)&rpmsg->link_state == 0U)
        {
            /* ~30 ms hand-rolled busy-wait */
            for (volatile uint32_t i = 0U; i < 16000000U; i++) {}
            PRINTF("  poll#%u link_state=%u\r\n", ++poll, *(volatile uint32_t *)&rpmsg->link_state);
        }
    }
    PRINTF("Link up!\r\n");

    PRINTF("Step 1: rpmsg_queue_create\r\n");
    rpmsg_queue_handle queue = rpmsg_queue_create(rpmsg);
    PRINTF("Step 2: queue=0x%08X\r\n", (uint32_t)queue);
    if (queue == RL_NULL) { PRINTF("ERROR: queue alloc failed\r\n"); for (;;) {} }

    PRINTF("Step 3: rpmsg_lite_create_ept (addr=%u)\r\n", LOCAL_EPT_ADDR);
    struct rpmsg_lite_endpoint *ept = rpmsg_lite_create_ept(
        rpmsg, LOCAL_EPT_ADDR, rpmsg_queue_rx_cb, queue);
    PRINTF("Step 4: ept=0x%08X\r\n", (uint32_t)ept);
    if (ept == RL_NULL) { PRINTF("ERROR: ept alloc failed\r\n"); for (;;) {} }

    PRINTF("Step 5: delaying 1s before announce\r\n");
    SDK_DelayAtLeastUs(1000000U, SDK_DEVICE_MAXIMUM_CPU_CLOCK_FREQUENCY);

    PRINTF("Step 6: rpmsg_ns_announce '%s'\r\n", RPMSG_LITE_NS_ANNOUNCE_STRING);
    int32_t announce_rc = rpmsg_ns_announce(rpmsg, ept, RPMSG_LITE_NS_ANNOUNCE_STRING, RL_NS_CREATE);
    PRINTF("Step 7: announce rc=%d — waiting for A53 handshake\r\n", announce_rc);

    /* Block until Linux sends the handshake byte — also captures A53 endpoint addr. */
    volatile uint32_t remote_addr = 0U;
    char              handshake[4];
    rpmsg_queue_recv(rpmsg, queue, (uint32_t *)&remote_addr,
                     handshake, sizeof(handshake), NULL, RL_BLOCK);
    PRINTF("A53 connected at addr %u. Streaming...\r\n", remote_addr);

    uint32_t seq_idx     = 0U;
    uint32_t state_count = 0U;
    PRINTF("[%s]\r\n", names[seq[seq_idx]]);

    while (1)
    {
        accel_state_t state = seq[seq_idx];
        accel_msg_t   msg   = make_sample(state);

        PRINTF("[%s] x=%6d y=%6d z=%6d\r\n", names[state], msg.x, msg.y, msg.z);
        (void)rpmsg_lite_send(rpmsg, ept, remote_addr,
                              (char *)&msg, sizeof(msg), RL_BLOCK);

        vTaskDelay(pdMS_TO_TICKS(SAMPLE_DELAY_MS));

        if (++state_count >= limits[seq_idx])
        {
            state_count = 0U;
            seq_idx     = (seq_idx + 1U) % 4U;
            PRINTF("\r\n[%s]\r\n", names[seq[seq_idx]]);
        }
    }
}

/*******************************************************************************
 * Main
 ******************************************************************************/
int main(void)
{
    BOARD_InitHardware();

    if (xTaskCreate(accel_task, "ACCEL", APP_TASK_STACK_SIZE, NULL,
                    tskIDLE_PRIORITY + 1U, NULL) != pdPASS)
    {
        PRINTF("Failed to create task\r\n");
        for (;;) {}
    }

    vTaskStartScheduler();

    PRINTF("Failed to start FreeRTOS\r\n");
    for (;;) {}
}
