
#include <stdio.h>
#include <npp.h>
#include <nppdefs.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <nppi_geometry_transforms.h>
#include <nppi_support_functions.h>


#include <iostream>
#include <nppi.h>

/*
    NppiInterpolationMode eInterploationMode = NPPI_INTER_SUPER;
    if (nScaleFactor >= 1.f) eInterploationMode = NPPI_INTER_LANCZOS;
    NPP_CHECK_NPP(nppiResize_8u_C1R(..., eInterploationMode));
*/

int main()
{
NppiInterpolationMode eInterploationMode = NPPI_INTER_SUPER;
int image_a_pitch;
NppiSize image_a_size = {.width = 960, .height = 540};
NppiRect image_a_roi = {.x = 0, .y = 0, .width = 960, .height = 540};
Npp8u* image_a = nppiMalloc_8u_C3(960, 540, &image_a_pitch);

int image_b_pitch;
NppiSize image_b_size = {.width = 480, .height = 270};
NppiRect image_b_roi = {.x = 0, .y = 0, .width = 480, .height = 270};
Npp8u* image_b = nppiMalloc_8u_C3(480, 270, &image_b_pitch);

NppStatus result = nppiResize_8u_C3R(image_a, image_a_pitch, image_a_size, image_a_roi, image_b, image_b_pitch, image_b_size, image_b_roi, NPPI_INTER_LANCZOS);

if (result != NPP_SUCCESS) {
    std::cerr << "Error executing Resize -- code: " << result << std::endl;
}
}
