cmake_minimum_required(VERSION 3.22.0)

include(${SdkRootDirPath}/cmake/extension/mcux.cmake)

project(rpmsg_accel_linux_remote LANGUAGES C CXX ASM
        PROJECT_BOARD_PORT_PATH ${board_root}/${board}/multicore_examples/rpmsg_accel_linux/remote)

include(${SdkRootDirPath}/CMakeLists.txt)

mcux_add_source(
    BASE_PATH ${SdkRootDirPath}
    SOURCES examples/multicore_examples/rpmsg_accel_linux/remote/main.c
)

mcux_add_include(
    BASE_PATH ${SdkRootDirPath}
    INCLUDES examples/multicore_examples/rpmsg_accel_linux/remote
             ${board_root}/${board}/multicore_examples/rpmsg_accel_linux/remote
             ${board_root}/${board}/multicore_examples/rpmsg_accel_linux/remote/${core_id}
)

mcux_add_macro(
    CC "-DPRINTF_ADVANCED_ENABLE=1"
)

mcux_add_linker_symbol(
    SYMBOLS "__stack_size__=0x400 __use_shmem__=1 __multicore__=1"
)

include(${SdkRootDirPath}/${board_root}/${board}/multicore_examples/rpmsg_accel_linux/remote/reconfig.cmake OPTIONAL)

mcux_convert_binary(BINARY ${APPLICATION_BINARY_DIR}/${MCUX_SDK_PROJECT_NAME}.bin)
