// Fake sys/socket.h to appease cuFile testing on Windows
#pragma once

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    
    // Sometimes Windows needs this for types like ssize_t
    #include <BaseTsd.h> 
    typedef SSIZE_T ssize_t;
#else
    // On Linux, bypass this shim and load the real system header
    #include_next <sys/socket.h>
#endif
