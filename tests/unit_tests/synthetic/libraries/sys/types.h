// Fake sys/types.h to appease cuFile testing on Windows
#pragma once

// ALWAYS load the real system header first (works for Windows SDK and Linux glibc)
#include_next <sys/types.h>

// 2. Add our Linux missing types
#ifdef _WIN32
    #include <stdint.h>
    
    // Define the Linux 64-bit file offset type for Windows
    typedef int64_t loff_t;
    typedef int64_t off64_t; 
    
    // (Optional) Include corecrt just in case standard sys/types.h misses it
    #include <corecrt.h> 
#endif