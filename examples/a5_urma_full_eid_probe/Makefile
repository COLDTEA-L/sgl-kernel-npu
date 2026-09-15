CC ?= gcc
UMDK_ROOT ?= /home/l00934901/umdk
URMA_INCLUDE ?= $(UMDK_ROOT)/src/urma/lib/urma/core/include
URMA_LIB_DIR ?= /usr/lib64

CFLAGS := -std=c11 -O2 -Wall -Wextra -Werror -I$(URMA_INCLUDE)
LDFLAGS := -L$(URMA_LIB_DIR) -Wl,-rpath,$(URMA_LIB_DIR) -lurma -lpthread

.PHONY: all clean
all: resolve_full_eid

resolve_full_eid: resolve_full_eid.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f resolve_full_eid
