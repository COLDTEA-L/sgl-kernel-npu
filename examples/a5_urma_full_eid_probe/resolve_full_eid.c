/* SPDX-License-Identifier: MIT */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "urma_api.h"

static int eid_equal(const urma_eid_t *lhs, const urma_eid_t *rhs)
{
    return memcmp(lhs, rhs, sizeof(*lhs)) == 0;
}

int main(int argc, char **argv)
{
    urma_eid_t requested = {0};
    urma_device_t *dev = NULL;
    urma_eid_info_t *list = NULL;
    urma_context_t *ctx = NULL;
    uint32_t count = 0;
    uint32_t eid_index = UINT32_MAX;
    int rc = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s FULL_EID\n", argv[0]);
        return 2;
    }
    if (urma_str_to_eid(argv[1], &requested) != 0) {
        fprintf(stderr, "EID_RESULT status=INVALID eid=%s\n", argv[1]);
        return 2;
    }
    if (urma_init(NULL) != URMA_SUCCESS) {
        fprintf(stderr, "EID_RESULT status=INIT_FAILED eid=%s errno=%d\n", argv[1], errno);
        return 3;
    }

    dev = urma_get_device_by_eid(requested, URMA_TRANSPORT_UB);
    if (dev == NULL) {
        printf("EID_RESULT status=NOT_VISIBLE eid=%s\n", argv[1]);
        rc = 4;
        goto out;
    }
    list = urma_get_eid_list(dev, &count);
    for (uint32_t i = 0; list != NULL && i < count; ++i) {
        if (eid_equal(&requested, &list[i].eid)) {
            eid_index = list[i].eid_index;
            break;
        }
    }
    if (eid_index == UINT32_MAX) {
        printf("EID_RESULT status=INDEX_NOT_FOUND eid=%s device=%s\n", argv[1], dev->name);
        rc = 5;
        goto out;
    }

    ctx = urma_create_context(dev, eid_index);
    if (ctx == NULL) {
        printf("EID_RESULT status=CONTEXT_FAILED eid=%s device=%s eid_index=%u errno=%d\n",
               argv[1], dev->name, eid_index, errno);
        rc = 6;
        goto out;
    }
    if (!eid_equal(&requested, &ctx->eid)) {
        printf("EID_RESULT status=CONTEXT_MISMATCH eid=%s device=%s eid_index=%u\n",
               argv[1], dev->name, eid_index);
        rc = 7;
        goto out;
    }

    printf("EID_RESULT status=PASS eid=%s device=%s eid_index=%u context_verified=1\n",
           argv[1], dev->name, eid_index);
    rc = 0;

out:
    if (ctx != NULL) {
        (void)urma_delete_context(ctx);
    }
    if (list != NULL) {
        urma_free_eid_list(list);
    }
    (void)urma_uninit();
    return rc;
}
