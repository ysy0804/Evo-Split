#ifndef EDGE_AI_WIN32_SOCKET_H
#define EDGE_AI_WIN32_SOCKET_H

#ifdef _WIN32
// Windows: use WinSock2
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

// POSIX compatibility macros for WinSock
#define SHUT_RDWR SD_BOTH
#define close(fd)        closesocket(fd)

// Windows setsockopt takes const char* for optval
#define setsockopt(s, l, o, v, l2) setsockopt((s), (l), (o), (const char*)(v), (l2))
// Windows: inet_ntop via inet_ntoa (address already in network byte order)
inline const char* inet_ntop(int af, const void* src, char* dst, int cnt) {
    (void)af; (void)cnt;
    strcpy(dst, inet_ntoa(*(const struct in_addr*)src));
    return dst;
}
// Windows: inet_pton via inet_addr
inline int inet_pton(int af, const char* src, void* dst) {
    if (af != AF_INET) return 0;
    struct in_addr addr;
    addr.s_addr = inet_addr(src);
    if (addr.s_addr == INADDR_NONE) return 0;
    *(struct in_addr*)dst = addr;
    return 1;
}

// WinSock init (call once at program start)
inline void net_init() {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
}
inline void net_cleanup() { WSACleanup(); }

#else
// Linux/Android: standard POSIX
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>

inline void net_init() {}
inline void net_cleanup() {}
#endif

// netRead / netWrite — portable read/write on sockets (avoid read/write macros
// which conflict with C++ iostream). Defined outside #ifdef so both Windows
// and POSIX builds can use them.
inline int netRead(int fd, void* buf, size_t n) {
#ifdef _WIN32
    return recv(fd, (char*)buf, (int)n, 0);
#else
    return (int)::read(fd, buf, n);
#endif
}
inline int netWrite(int fd, const void* buf, size_t n) {
#ifdef _WIN32
    return send(fd, (const char*)buf, (int)n, 0);
#else
    return (int)::write(fd, buf, n);
#endif
}

#endif // EDGE_AI_WIN32_SOCKET_H
