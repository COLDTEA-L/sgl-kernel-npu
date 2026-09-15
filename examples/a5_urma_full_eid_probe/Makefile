CC ?= gcc
UMDK_ROOT ?= /home/l00934901/umdk
URMA_HEADER := $(firstword $(wildcard \
	/usr/include/urma/urma_api.h \
	/usr/local/include/urma/urma_api.h \
	/usr/include/urma_api.h \
	/usr/local/include/urma_api.h \
	$(UMDK_ROOT)/src/urma/lib/urma/core/include/urma_api.h))
URMA_INCLUDE ?= $(patsubst %/urma_api.h,%,$(URMA_HEADER))
URMA_LIB_DIR ?= /lib64

CFLAGS := -std=c11 -O2 -Wall -Wextra -Werror -I$(URMA_INCLUDE)
LDFLAGS := -L$(URMA_LIB_DIR) -Wl,-rpath,$(URMA_LIB_DIR) -lurma -lpthread

.PHONY: all check-env clean
all: check-env resolve_full_eid

check-env:
	@test -n "$(URMA_INCLUDE)" -a -f "$(URMA_INCLUDE)/urma_api.h" || { \
		echo "ERROR: urma_api.h not found." >&2; \
		echo "Checked /usr/include/urma, /usr/local/include/urma and $(UMDK_ROOT)/src/urma/lib/urma/core/include." >&2; \
		echo "Pass URMA_INCLUDE=/actual/header/directory to make." >&2; \
		exit 1; \
	}
	@test -e "$(URMA_LIB_DIR)/liburma.so" -o -e "$(URMA_LIB_DIR)/liburma.so.0" || { \
		echo "ERROR: liburma.so not found under $(URMA_LIB_DIR)." >&2; \
		echo "Run: ldconfig -p | grep liburma" >&2; \
		exit 1; \
	}
	@echo "URMA include: $(URMA_INCLUDE)"
	@echo "URMA library: $(URMA_LIB_DIR)"

resolve_full_eid: resolve_full_eid.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f resolve_full_eid
