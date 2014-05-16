/*
 * config.c
 *
 *  Created on: Nov 12, 2013
 *      Author: tcarstens
 */


#include "state.h"

struct config_t g_config =
{
	.config_platform = 1,
	.config_device = 0,
	.config_mp_source = "mp.cl",
#ifdef NVIDIA_OUTPUT_PTX
	.config_mp_binary = "mp.S",
#endif /* NVIDIA_OUTPUT_PTX */
	.config_global_work_size = 1024*1024
};
