// course/lanes.rs: the world position W(distance, lane) on the course path, from the race's block of the course
// buffer (gpu::pack::course_words, at r_course_base()). Quaternions are vec4 (x, y, z, w); every product and sum is
// written out in the Rust order (no dot(), no vector arithmetic). Needs exact.glsl and race_state.glsl.

#ifndef UMA_COURSE_PATH_GLSL
#define UMA_COURSE_PATH_GLSL

#include "course_buffer.glsl"

// x % y for doubles (Rust f64 %, C fmod): exact. Each step subtracts the largest y * 2^k that fits, which is exact
// (Sterbenz: s <= r < 2s).
double fmod64(double x, double y) {
    if (is_nan64(x) || is_nan64(y) || !is_finite64(x) || is_zero64(y)) return uint64BitsToDouble(0x7ff8000000000000UL);
    if (!is_finite64(y)) return x;
    double ax = abs(x), ay = abs(y);
    if (ax < ay) return x;
    double r = ax;
    while (r >= ay) {
        double s = ay;
        while (s * 2.0lf <= r) s = s * 2.0lf;
        r = r - s;
    }
    return sign_bit64(x) ? neg_bits64(r) : r;
}

float quat_dot(vec4 a, vec4 b) { return ((a.x * b.x + a.y * b.y) + a.z * b.z) + a.w * b.w; }

// Quaternion::lerp: normalized, towards -b when the dot is negative; plain f32.
vec4 quat_lerp(vec4 a, vec4 b, float t) {
    t = mathf_clamp01(t);
    vec4 q;
    if (quat_dot(a, b) < 0.0) {
        q.x = a.x + t * (-b.x - a.x);
        q.y = a.y + t * (-b.y - a.y);
        q.z = a.z + t * (-b.z - a.z);
        q.w = a.w + t * (-b.w - a.w);
    } else {
        q.x = a.x + t * (b.x - a.x);
        q.y = a.y + t * (b.y - a.y);
        q.z = a.z + t * (b.z - a.z);
        q.w = a.w + t * (b.w - a.w);
    }
    float m = sqrt_rn32(quat_dot(q, q));
    return vec4(div_rn32(q.x, m), div_rn32(q.y, m), div_rn32(q.z, m), div_rn32(q.w, m));
}

// Quaternion::mul: each component one f64 expression.
vec4 quat_mul(vec4 a, vec4 b) {
    double ax = a.x, ay = a.y, az = a.z, aw = a.w, bx = b.x, by = b.y, bz = b.z, bw = b.w;
    return vec4(
        float(((aw * bx + ax * bw) + ay * bz) - az * by),
        float(((aw * by + ay * bw) + az * bx) - ax * bz),
        float(((aw * bz + az * bw) + ax * by) - ay * bx),
        float(((aw * bw - ax * bx) - ay * by) - az * bz));
}

// Quaternion::rotate: f32 temporaries, each component one f64 expression.
vec3 quat_rotate(vec4 q, vec3 p) {
    float x = q.x + q.x, y = q.y + q.y, z = q.z + q.z;
    double xx = q.x * x, yy = q.y * y, zz = q.z * z, xy = q.x * y, xz = q.x * z, yz = q.y * z;
    double wx = q.w * x, wy = q.w * y, wz = q.w * z;
    double px = p.x, py = p.y, pz = p.z;
    return vec3(
        float(((1.0lf - (yy + zz)) * px + (xy - wz) * py) + (xz + wy) * pz),
        float(((xy + wz) * px + (1.0lf - (xx + zz)) * py) + (yz - wx) * pz),
        float(((xz - wy) * px + (yz + wx) * py) + (1.0lf - (xx + yy)) * pz));
}

// `(b - a) * frac + a` in f64, rounded once.
float anim_lerp(float a, float b, float frac) { return float((double(b) - double(a)) * double(frac) + double(a)); }

// lanes::sample on the animation block at word `a`.
void anim_sample(uint a, float t, out vec3 pos, out vec4 rot) {
    uint lx = crs[a + 1u], ly = crs[a + 2u], lz = crs[a + 3u], lr = crs[a + 4u];
    uint X = a + 5u, Y = X + lx, Z = Y + ly, R = Z + lz;
    int i = f32_to_i32(t);
    float frac = t - float(i);
    bool has_y = ly != 0u;
    bool in_range = i >= 0 && uint(i) < lx && uint(i) < lz && (!has_y || uint(i) < ly);
    if (!in_range) {
        pos = vec3(0.0);
    } else if (uint(i) < lx - 1u) {
        uint i0 = uint(i), i1 = uint(i) + 1u;
        float f = mathf_clamp01(frac);
        pos.x = anim_lerp(crs_f(X + i0), crs_f(X + i1), f);
        pos.y = has_y ? anim_lerp(crs_f(Y + i0), crs_f(Y + i1), f) : anim_lerp(0.0, 0.0, f);
        pos.z = anim_lerp(crs_f(Z + i0), crs_f(Z + i1), f);
    } else {
        uint n = lx - 1u;
        pos = vec3(crs_f(X + n), has_y ? crs_f(Y + n) : 0.0, crs_f(Z + n));
    }
    if (i < int(lr) - 1) {
        if (i < 0) {
            set_r_gpu_fault(r_gpu_fault() | 64u);  // unreachable for t in [0, 1000] (Rust would panic)
            rot = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        uint r0 = R + uint(i) * 4u;
        vec4 qa = vec4(crs_f(r0), crs_f(r0 + 1u), crs_f(r0 + 2u), crs_f(r0 + 3u));
        vec4 qb = vec4(crs_f(r0 + 4u), crs_f(r0 + 5u), crs_f(r0 + 6u), crs_f(r0 + 7u));
        rot = quat_lerp(qa, qb, t - float(i));
    } else {
        uint r0 = R + (lr - 1u) * 4u;
        rot = vec4(crs_f(r0), crs_f(r0 + 1u), crs_f(r0 + 2u), crs_f(r0 + 3u));
    }
}

// CourseLane::lane_transform: lane-0 position and rotation at `distance`.
void lane_transform(float distance, out vec3 pos, out vec4 rot) {
    uint base = r_course_base();
    uint main_a = base + crs[base], over = crs[base + 1u];
    float len = float(int(crs[main_a]));
    float t;
    uint a;
    if (len >= distance) {
        a = main_a;
        t = float(div_rn64(double(distance), double(len)) * 1000.0lf);
    } else if (over != 0u) {
        a = base + over;
        float olen = float(int(crs[a]));
        t = float(div_rn64(fmod64(double(distance) - double(len), double(olen)), double(olen)) * 1000.0lf);
    } else {
        float d = distance - len;
        pos = vec3(crs_f(base + 5u) * d + crs_f(base + 2u), crs_f(base + 6u) * d + crs_f(base + 3u), crs_f(base + 7u) * d + crs_f(base + 4u));
        rot = vec4(crs_f(base + 8u), crs_f(base + 9u), crs_f(base + 10u), crs_f(base + 11u));
        return;
    }
    vec4 q;
    anim_sample(a, t, pos, q);
    rot = quat_mul(q, vec4(crs_f(base + 12u), crs_f(base + 13u), crs_f(base + 14u), crs_f(base + 15u)));
}

// lanes::calc_world_transform(...).pos_world: the lane-0 origin offset sideways by lane_distance * 11.25.
vec3 world_position(float distance, float lane_distance) {
    vec3 origin;
    vec4 rot;
    lane_transform(distance, origin, rot);
    vec3 r = quat_rotate(rot, vec3(lane_distance * 11.25, 0.0, 0.0));
    return vec3(r.x + origin.x, r.y + origin.y, r.z + origin.z);
}

#endif
