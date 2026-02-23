// Fake arpa/inet.h to appease cuFile testing on Windows
#pragma once

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
#else
    // On Linux, bypass this shim and load the real system header
    #include_next <arpa/inet.h>
#endif
