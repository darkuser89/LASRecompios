// Temporary shared file backing for the compact guest's CPU resource cache. Unlike anonymous
// allocator pages, inactive archive data can be written back and reclaimed by the OS. The file is
// private and unlinked immediately; its storage disappears when the mapping is released.
#pragma once
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

// A fixed address is only valid inside an already-owned guest reservation, before any guest/GPU use.
static inline void *resource_cache_map_at(void *address, size_t size) {
    const char *dir = getenv("TMPDIR");
    if (!dir || !*dir) dir = "/tmp";
    char path[1024];
    int n = snprintf(path, sizeof path, "%s%slas-resource-XXXXXX", dir, dir[strlen(dir) - 1] == '/' ? "" : "/");
    if (n < 0 || (size_t)n >= sizeof path) { errno = ENAMETOOLONG; return NULL; }
    int fd = mkstemp(path);
    if (fd < 0) return NULL;
    if (unlink(path) != 0 || ftruncate(fd, (off_t)size) != 0) {
        int error = errno;
        close(fd);
        unlink(path);
        errno = error;
        return NULL;
    }
    void *p = mmap(address, size, PROT_READ | PROT_WRITE, MAP_SHARED | (address ? MAP_FIXED : 0), fd, 0);
    int error = errno;
    close(fd);
    if (p == MAP_FAILED) { errno = error; return NULL; }
    return p;
}

static inline void *resource_cache_map(size_t size) { return resource_cache_map_at(NULL, size); }
