# Deviations from Toradex / NXP Documentation

Running log of things that had to be done differently from what the official
documentation describes, discovered while building the TCC RPMsg firmware for
Verdin iMX8MP (MIMX8ML8) on Dahlia carrier board, TorizonOS.

---

## 1. `bootaux` requires a `.bin` file, not `.elf`

**What the docs imply:** Copy the compiled firmware image to ITCM and run
`bootaux`. The example uses a path to the firmware file without specifying
format.

**What actually happens:** Loading an `.elf` file with `bootaux` causes an
immediate "Synchronous Abort" in U-Boot and a board reset. `bootaux` reads the
bytes at the load address, sees the ELF magic (`0x7F 45 4C 46`), and attempts
to parse ELF segment headers. Those headers contain virtual addresses in A53
space that the bootloader then tries to access, triggering the abort.

**Fix:** Copy the `.bin` output (raw binary), not the `.elf`, to the board.
The full `cm_boot` sequence:
```
ext4load mmc 0:1 ${loadaddr} /path/to/firmware.bin
cp.b ${loadaddr} 0x7e0000 ${cm_isize}
dcache flush
mw.w 0x550ff000 0 64
bootaux 0x7e0000
```

---

## 2. M7 must start via U-Boot before Linux boots

**What the docs describe:** Both early boot (U-Boot `bootaux`) and late load
(Linux remoteproc: `echo start > /sys/class/remoteproc/remoteproc0/state`) are
listed as valid options.

**What actually happens:** Starting M7 via remoteproc (after Linux is already
running) causes the A53 to freeze hard. Root cause: `BOARD_InitHardware()` →
`BOARD_BootClockRUN()` reconfigures PLLs and clock roots that Linux is actively
using. Calling it after Linux is up corrupts the running system.

**Fix:** Always use the U-Boot early-boot path (`run cm_boot`) so M7 starts
before Linux initialises clocks. `BOARD_BootClockRUN()` is safe only when the
A53 is still in U-Boot.

---

## 3. Channel name must be `"rpmsg-tty"` to get `/dev/ttyRPMSG0`

**What the NXP SDK example uses:** The `rpmsg_lite_str_echo_rtos` example sets
`RPMSG_LITE_NS_ANNOUNCE_STRING` to `"rpmsg-virtual-tty-channel-1"`.

**What TorizonOS actually expects:** The TorizonOS kernel ships the
`rpmsg_tty` module (alias `rpmsg:rpmsg-tty`). This module only creates
`/dev/ttyRPMSG0` when it sees a name-service announcement with the channel name
`"rpmsg-tty"`. Any other name is silently ignored — no device node appears,
no error in dmesg.

**Fix:** In `app.h`:
```c
#define RPMSG_LITE_NS_ANNOUNCE_STRING "rpmsg-tty"
```

Also required: `#define RPMSG_LITE_MASTER_IS_LINUX` to use the Linux-compatible
virtio memory layout.

---

## 4. False "timeout" in link-up wait — `RL_SUCCESS == 0` collides with timeout return value

**Status:** Fixed 2026-06-05.

**Symptom:** M7 firmware hung silently after printing `MUB SR=... CR=...`.
Linux showed `rpmsg host is online` but no channel was ever created.

**Root cause:** Linux kicked M7 (wrote to MUA_TR1) during the window inside
`rpmsg_lite_remote_init()` between `MU_EnableInterrupts()` and the first
`env_disable_interrupt()`. The MU ISR fired, read `MUB_RR1`, called
`rpmsg_lite_tx_callback()`, and set `link_state = 1` — all before
`rpmsg_lite_remote_init()` returned. The link was already up.

`rpmsg_lite_wait_for_link_up()` saw `link_state != 0` and returned
`RL_SUCCESS = 0` immediately. The diagnostic check
`if (link_result == 0U)` treated `RL_SUCCESS = 0` as a timeout (same numeric
value), hung in `for(;;)`, and M7 never sent the NS announcement.

**Confirmed by:** `MUB CR = 0x04000000` (bit 26 = RIE1 set ✓) and
`MUB SR = 0x00F00000` (RF1 = 0, meaning MUB_RR1 was already read and cleared
by the ISR before the print).

**Fix:** Check `rpmsg->link_state == 0U` after the wait instead of the
return value of `rpmsg_lite_wait_for_link_up()`. The return value conflates
"link already up when called" (`RL_SUCCESS = 0`) with "timed out" (also 0).

```c
(void)rpmsg_lite_wait_for_link_up(rpmsg, 10000U);
if (rpmsg->link_state == 0U)
{
    /* True timeout — Linux never sent the MU kick */
    for (;;) {}
}
```

---

## 5. `rpmsg_tty` is not autoloaded — must `modprobe` manually

**What the docs imply:** Once M7 announces the `"rpmsg-tty"` channel via the
name service, udev matches the `rpmsg:rpmsg-tty` modalias, loads `rpmsg_tty.ko`
automatically, and `/dev/ttyRPMSG0` appears without any extra steps.

**What actually happens on TorizonOS:** The NS announcement reaches Linux
correctly (`virtio_rpmsg_bus virtio0: creating channel rpmsg-tty addr 0x1e`
appears in dmesg), but `rpmsg_tty` is never probed. No udev rule triggers the
`modprobe` and no `/dev/ttyRPMSG0` node is created.

**Fix:** Load the module manually after boot:
```bash
sudo modprobe rpmsg_tty
```
`/dev/ttyRPMSG0` appears immediately. For persistence across reboots, add the
module to the host autoload list:
```bash
echo rpmsg_tty | sudo tee -a /etc/modules
```

---

## 6. Linux must write first to `/dev/ttyRPMSG0` before M7 starts streaming

**What the docs imply:** After the channel is created, the application on A53
simply opens `/dev/ttyRPMSG0` and reads the stream.

**What actually happens:** After the NS announce, M7 calls
`rpmsg_queue_recv(rpmsg, queue, &remote_addr, ..., RL_BLOCK)` before sending
any data. This is required — RPMsg-Lite does not expose the Linux-side endpoint
address via any API. The only way M7 learns the return address is from the `src`
field of the first received message. Until Linux writes something to the tty,
M7 blocks at the `recv` and produces no output.

The standard `rpmsg_tty` driver does NOT send any "hello" on probe or open;
the first write must come from userspace.

**Fix:** Write at least one byte to the device before reading:
```bash
printf '\x00' | sudo tee /dev/ttyRPMSG0 > /dev/null   # unblocks M7
sudo cat /dev/ttyRPMSG0 | xxd                          # read the accel stream
```
In the Python receiver script, open the device for writing first, write a kick
byte, then read in a loop.

---

## 7. `clk_ignore_unused` required in kernel cmdline

**What the docs imply:** Disabling unused kernel clocks is a default Linux
behaviour that should not affect M7 firmware running in bare-metal mode.

**What actually happens:** Linux's clock management disables clocks that no
kernel driver has claimed. When UART4 (and other M7-facing peripherals) are
suppressed by device-tree overlays, no Linux driver claims their clocks. Linux
then gates those clocks during boot, starving M7 peripherals.

**Fix:** Add `clk_ignore_unused` to the kernel command line in U-Boot:
```
setenv bootargs ${bootargs} clk_ignore_unused
saveenv
```
Confirmed working — dmesg shows `clk: Not disabling unused clocks` at boot.

---

## 8. UART4 imx-uart driver binds despite disable overlay (cosmetic)

**What the docs imply:** Adding a device-tree overlay that sets
`status = "disabled"` on the UART4 node prevents any Linux driver from
claiming it, keeping the UART free for M7 debug output.

**What actually happens:** Despite the overlay, the imx-uart driver still
probes UART4 (`30860000.serial: ttymxc0 at MMIO 0x30860000 is a IMX` in
dmesg at ~0.987 s). The driver resets UART hardware state and M7's debug UART
output stops at that point.

**Impact:** Cosmetic only. M7 firmware continues executing silently — the NS
announcement arriving in dmesg at 2.545 s proves M7 ran past UART death. All
runtime functionality works correctly; you simply lose M7 serial traces after
kernel start.

**Workaround:** Either accept the log blackout (M7 still works), or use a more
aggressive overlay strategy (e.g., setting `pinctrl` mux to GPIO on UART4 pins
before the kernel takes over, preventing the UART driver from finding a live
peripheral).

---

## Reference: Working `cm_boot` U-Boot environment variable

```
cm_boot=ext4load mmc 0:1 ${loadaddr} /lib/firmware/rpmsg_accel_linux.bin; \
        cp.b ${loadaddr} 0x7e0000 ${cm_isize}; \
        dcache flush; \
        mw.w 0x550ff000 0 64; \
        bootaux 0x7e0000
```

Set once via `setenv cm_boot ...` then `saveenv`.
