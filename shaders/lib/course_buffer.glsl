// The course buffer (gpu::pack::course_words): one block per course, the race's at r_course_base().

#ifndef UMA_COURSE_BUFFER_GLSL
#define UMA_COURSE_BUFFER_GLSL

#ifndef COURSE_BINDING
#define COURSE_BINDING 2
#endif

layout(std430, binding = COURSE_BINDING) readonly buffer Course { uint crs[]; };

float crs_f(uint at) { return uintBitsToFloat(crs[at]); }

#endif
