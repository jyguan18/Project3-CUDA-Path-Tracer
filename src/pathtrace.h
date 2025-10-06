#pragma once

#include "scene.h"
#include "utilities.h"
#include "../external/OpenImageDenoise/include/oidn.hpp"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
