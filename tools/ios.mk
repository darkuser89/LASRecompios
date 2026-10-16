# iPadOS build (make ios-*). Included by the Makefile; see docs/ios.md.
#
# Three things differ from the desktop builds:
#   - a compact guest window and smaller host thread stacks fit the normal iPadOS address space. Module bases
#     stay identical to desktop, so the generated game code can be shared.
#   - everything is static: SDL3 + FFmpeg from tools/build_deps_ios.sh (no dylibs to ship), and the game code is
#     linked into the app (liblasgame_ios.a) instead of being built on the device by a compiler the app spawns.
#   - Xcode does the runtime itself (ios/project.yml through xcodegen) so it can sign, install and debug it.

IOS_MIN       ?= 17.0
# The guest window stays at the upstream base on iOS: the kernel itself places large anonymous allocations at
# 0x7000000000 (measured on the Air 4 for 2-4 GB allocations). The compact reservation is 2.5625 GB;
# ios/project.yml enables it without requiring an Extended Virtual Addressing entitlement.
# Because the base is the upstream one, the iOS build shares generated/ and the game code with the desktop build.
IOS_BASE_MAIN ?= 0x7000000000
IOS_BASE_SDK  ?= 0x7004000000
IOS_GEN       ?= generated
IOS_B         := build/ios
IOS_UDID      ?= 00008112-000924EE14D9A01E   # iPad Air 13" (M2); override for another device
IOS_TEAM      ?= K5TW9AU245                  # Apple Development team (Konrad Kern)
IOS_BUNDLE_ID ?= com.konradkern.lasrecomp
IOS_APP       := $(IOS_B)/dd/Build/Products/Debug-iphoneos/LAS Recomp.app

# xcode-select on this machine points at the Command Line Tools; DEVELOPER_DIR picks Xcode without sudo.
IOS_DEV_DIR ?= $(shell [ -d /Applications/Xcode.app/Contents/Developer ] && echo /Applications/Xcode.app/Contents/Developer || xcode-select -p)
IOS_XCRUN   := DEVELOPER_DIR=$(IOS_DEV_DIR) xcrun
IOS_SDK     := $(shell DEVELOPER_DIR=$(IOS_DEV_DIR) xcrun --sdk iphoneos --show-sdk-path 2>/dev/null)
IOS_TARGET  := arm64-apple-ios$(IOS_MIN)

IOS_GENFLAGS := -O1 -fno-strict-aliasing -Wno-return-type -Wno-unused-label -fvisibility=hidden -Iruntime \
                -target $(IOS_TARGET) -isysroot $(IOS_SDK)
IOS_GEN_SRC  := $(wildcard $(IOS_GEN)/main/*.c) $(wildcard $(IOS_GEN)/sdk/*.c)
IOS_GEN_OBJ  := $(patsubst %.c,$(IOS_B)/obj/%.o,$(patsubst $(IOS_GEN)/%,%,$(IOS_GEN_SRC))) \
                $(IOS_B)/obj/tools/gamelib_entry.o

.PHONY: ios-deps ios-gen ios-game-lib ios-xcode ios-app ios-install ios-run ios-files ios-logs

# ---- the static SDL3 + FFmpeg the app links (pinned sources, LGPL FFmpeg: tools/build_deps_ios.sh)
ios-deps:
	tools/build_deps_ios.sh $(IOS_MIN)

# ---- the game code for iOS. Same guest base as the desktop build (IOS_BASE_MAIN), so this is only a freshness
# check: `make gen` fills generated/, and ios-game-lib compiles it for arm64-ios into build/ios/liblasgame_ios.a.
ios-gen:
	@[ -f extracted/exefs/main ] || { echo "no extracted/exefs/main: set up the game files first (tools/check_game_files.sh)"; exit 1; }
	@[ -d $(IOS_GEN)/main ] || $(MAKE) gen

$(IOS_B)/obj/%.o: $(IOS_GEN)/%.c
	@mkdir -p $(dir $@)
	$(IOS_XCRUN) --sdk iphoneos clang $(IOS_GENFLAGS) -c $< -o $@

$(IOS_B)/obj/tools/gamelib_entry.o: tools/gamelib_entry.c $(GAMELIB_SRCS)
	@mkdir -p $(dir $@)
	$(IOS_XCRUN) --sdk iphoneos clang $(IOS_GENFLAGS) -DGAMELIB_ID='"$(GAMELIB_ID)"' -c $< -o $@

# The SMAA shader include the runtime's smaa_data.c pulls in (the same generated file the desktop build uses).
ios-game-lib: $(IOS_GEN_OBJ) $(B)/inc/smaa_hlsl.inc
	@mkdir -p $(IOS_B)
	ar rcs $(IOS_B)/liblasgame_ios.a $(IOS_GEN_OBJ)   # (host ar: a plain arm64 Mach-O archive, no SDK tool needed)
	@echo "$(IOS_B)/liblasgame_ios.a: $$(ls -l $(IOS_B)/liblasgame_ios.a | awk '{printf "%.1f MB", $$5/1048576}'), $(words $(IOS_GEN_OBJ)) objects"

# ---- the Xcode project (xcodegen) and the app
# The app icon: the repo's 1024x1024 artwork, alpha flattened (iOS icons must be opaque), generated on demand.
IOS_ICON := ios/Assets.xcassets/AppIcon.appiconset/AppIcon1024.png
$(IOS_ICON): docs/icon/icon_1024.png
	@mkdir -p $(dir $@)
	$(PY) -c "from PIL import Image; im = Image.open('docs/icon/icon_1024.png').convert('RGBA'); \
	  bg = Image.new('RGB', im.size, (0, 0, 0)); bg.paste(im, mask=im.split()[3]); \
	  bg.resize((1024, 1024), Image.LANCZOS).save('$(IOS_ICON)')"
	@echo "$(IOS_ICON): $$(sips -g pixelWidth -g pixelHeight $(IOS_ICON) | tail -2 | tr -d ' \\n')"

ios-xcode: $(IOS_ICON)
	@command -v xcodegen >/dev/null || { echo "xcodegen is needed: brew install xcodegen"; exit 1; }
	rm -rf ios/LASRecomp.xcodeproj
	cd ios && xcodegen generate --spec project.yml   # next to the spec: its relative paths (Info.plist, entitlements) resolve

ios-app: ios-deps ios-game-lib ios-xcode
	DEVELOPER_DIR=$(IOS_DEV_DIR) xcodebuild -project ios/LASRecomp.xcodeproj -scheme LASRecomp \
	    -configuration Debug -destination 'generic/platform=iOS' -derivedDataPath $(IOS_B)/dd \
	    RT_BASE=$(IOS_BASE_MAIN) DEVELOPMENT_TEAM=$(IOS_TEAM) PRODUCT_BUNDLE_IDENTIFIER=$(IOS_BUNDLE_ID) \
	    CODE_SIGN_STYLE=Automatic -allowProvisioningUpdates build
	@echo "app: $(IOS_APP)"

# ---- install / run on the iPad (devicectl; Xcode's Run button does the same with a UI)
ios-install:
	$(IOS_XCRUN) devicectl device install app --device $(IOS_UDID) "$(IOS_APP)"

ios-run:
	$(IOS_XCRUN) devicectl device process launch --console --device $(IOS_UDID) $(IOS_BUNDLE_ID)

ios-logs:
	$(IOS_XCRUN) devicectl device process launch --console --terminate-existing --device $(IOS_UDID) $(IOS_BUNDLE_ID)

# ---- the player's own game files: exefs/, romfs.bin and the movies, straight into the app's Documents folder
# (the data root the iOS build uses, runtime/main.c root "."). The movies are loose files because the game's video
# decoder opens them by path (runtime/hle/movie.c) rather than through the romfs image; the image covers everything
# else. devicectl's --destination is the exact target path: each source is copied separately.
ios-files:
	@[ -d extracted/exefs ] || { echo "no extracted/exefs"; exit 1; }
	@[ -f extracted/romfs.bin ] || { echo "no extracted/romfs.bin: ./build/lasrecomp --build-romfs extracted/romfs extracted/romfs.bin"; exit 1; }
	@[ -d extracted/romfs/region_common/movie ] || { echo "no extracted/romfs/region_common/movie (the opening/title/ending videos)"; exit 1; }
	$(IOS_XCRUN) devicectl device copy to --device $(IOS_UDID) --domain-type appDataContainer \
	    --domain-identifier $(IOS_BUNDLE_ID) --source extracted/exefs --destination Documents/exefs
	$(IOS_XCRUN) devicectl device copy to --device $(IOS_UDID) --domain-type appDataContainer \
	    --domain-identifier $(IOS_BUNDLE_ID) --source extracted/romfs.bin --destination Documents/romfs.bin
	$(IOS_XCRUN) devicectl device copy to --device $(IOS_UDID) --domain-type appDataContainer \
	    --domain-identifier $(IOS_BUNDLE_ID) --source extracted/romfs/region_common/movie --destination Documents/romfs/region_common/movie
	@echo "pushed exefs/, romfs.bin and the movies into $(IOS_BUNDLE_ID)'s Documents"
