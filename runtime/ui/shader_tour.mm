// Shader preparation tour (first-launch shader compile, user 2026-10-08): LARECOMP_SHADER_TOUR=<frames per spot>
// warps through every warp spot of the game (runtime/ui/warp_data.h), letting each scene render for a moment so
// its shaders get translated and compiled - they land in cache/shaders/ and Metal's disk cache, which the boot
// warm-up (runtime/nvn/renderer.m shader_cache_warm) loads every later start: no first-use pop-in. Nothing derived
// from the game ships: the cache is built on the player's machine from their own copy. Exits when done.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "warp_data.h"

extern "C" {
#include "../hle.h"
int game_in_play(void);
void warp_request(const char *level, const char *locator);
int warp_busy(void);
bool warp_blocked(const char *locator);
unsigned renderer_pending_shaders(void);
void renderer_window_shot_begin(void);
int renderer_window_shot_visible(void);
extern char g_window_shot_request[256];
}

#define PLAYER_STATUS_PTR (MAIN_BASE + 0x1CC1368)  // PlayerStatus*: current level name at +0

extern "C" void shader_tour_tick(uint64_t frame) {
    static int per = -1, i, end, waiting;
    static uint64_t started, applied, deadline;
    const int n = (int)(sizeof kWarpSpots / sizeof kWarpSpots[0]);
    if (per < 0) {
        const char *e = ENV("LARECOMP_SHADER_TOUR"); per = e ? atoi(e) : 0;
        if (e && per < 120) per = 240;
        i = ENV("LARECOMP_TOUR_START") ? atoi(ENV("LARECOMP_TOUR_START")) : 0;
        end = ENV("LARECOMP_TOUR_END") ? atoi(ENV("LARECOMP_TOUR_END")) : n;
        if (i < 0) i = 0;
        if (end < 0 || end > n) end = n;
    }
    if (!per) return;
    if (i >= end && ENV("LARECOMP_TOUR_LOOP")) { i = 0; LOG("[tour] loop\n"); }
    if (i >= end) {
        LOG("[tour] done: through %d / %d at frame %llu\n", i, n, (unsigned long long)frame);
        exit(0);
    }
    uint64_t now = plat_now_ns();
    if (waiting && now > deadline) {
        LOG("[tour] not reached: %d / %d %s / %s\n", i, n, kWarpSpots[i].level, kWarpSpots[i].locator);
        if (ENV("LARECOMP_TOUR_FAIL_FAST")) exit(2);
        i++; waiting = 0; return;
    }
    if (!game_in_play()) { applied = 0; return; }
    static long from = ENV("LARECOMP_TOUR_FROM") ? atol(ENV("LARECOMP_TOUR_FROM")) : 0;
    if ((long)frame < from) return;
    if (!started) { started = frame; LOG("[tour] %d spots, %d settled frames each, range %d..%d\n", n, per, i, end); }
    static const char *only = ENV("LARECOMP_TOUR_LEVEL");
    static const char *pre = ENV("LARECOMP_TOUR_PREFIX");
    const WarpSpot &w = kWarpSpots[i];
    if (waiting == 2) {
        int visible = renderer_window_shot_visible();
        if (visible < 0) return;
        if (!visible) {
            LOG("[tour] black capture: %d / %d %s / %s; waiting for visible scene\n", i, n, w.level, w.locator);
            applied = 0;
            waiting = 1;
            return;
        }
        LOG("[tour] rendered: %d / %d %s / %s, visible capture after %llu settled frames, frame %llu\n",
            i, n, w.level, w.locator, (unsigned long long)(frame - applied), (unsigned long long)frame);
        i++; waiting = 0;
        return;
    }
    const char *skip = NULL;
    if ((only && strcmp(w.level, only)) || (pre && strncmp(w.locator, pre, strlen(pre)))) skip = "filter";
    if (ENV("LARECOMP_TOUR_PLAIN_STARTS") && !strncmp(w.locator, "Warp_", 5)) skip = "arrival-event start; use plain entry";
    if (warp_blocked(w.locator)) skip = "spawn precondition";
    if (skip) {
        LOG("[tour] skip: %d / %d %s / %s (%s)\n", i, n, w.level, w.locator, skip);
        i++; return;
    }
    if (!waiting) {
        LOG("[tour] visit: %d / %d %s / %s\n", i, n, w.level, w.locator);
        warp_request(w.level, w.locator);
        waiting = 1;
        applied = 0;
        deadline = now + 45000000000ull;
        return;
    }
    if (warp_busy()) { applied = 0; return; }
    uint64_t ps = LD64(PLAYER_STATUS_PTR);
    bool there = ps && !strcmp((const char *)(uintptr_t)ps, w.level);
    // Wait for compilation to finish, then render a complete stable interval before taking evidence.
    if (!there || renderer_pending_shaders()) { applied = 0; return; }
    if (!applied) applied = frame;
    if (frame - applied >= (uint64_t)per) {
        const char *shots = ENV("LARECOMP_TOUR_SHOTS");
        if (shots) {
            mkdir(shots, 0755);
            renderer_window_shot_begin();
            snprintf(g_window_shot_request, 256, "%s/%03d-%s-%s.png", shots, i, w.level, w.locator);
            waiting = 2;
            return;
        }
        LOG("[tour] rendered: %d / %d %s / %s, %llu settled frames, frame %llu\n", i, n, w.level, w.locator,
            (unsigned long long)(frame - applied), (unsigned long long)frame);
        i++;
        waiting = 0;
    }
}
