// util::intro_sort (the unstable .NET introsort) over indices sort_h[0..n), with the comparator chosen by
// sort_mode. Elements are horse indices (or, for SORT_KEY_TRUNC, indices into sort_key) and every comparator reads
// values the sort does not change, so this matches sorting the Rust values (a copied pivot compares like its
// element). Define SORT_CAP before including for more than MAX_HORSES elements. Needs race_state.glsl and
// exact.glsl.

#ifndef UMA_SORT_GLSL
#define UMA_SORT_GLSL

// ai/order.rs cmp_cur_order: distance descending (NaN last, as float.CompareTo), then horse index ascending.
const int SORT_CUR_ORDER = 0;
// field::calc_finish_order: finish time ascending (f64; NaN first, as double.CompareTo), then horse index.
const int SORT_FINISH_TIME = 1;
// cur_order ascending (wrapping difference).
const int SORT_ORDER = 2;
// field::update_overlap: (int)lane_distance ascending, as (a - b) * 10000 with wrapping.
const int SORT_LANE = 3;
// last_spurt.rs speed_down_next: sort_key ascending, as (int)((a - b) * 10000f) (truncating, not transitive).
const int SORT_KEY_TRUNC = 4;
// target.rs NearHorseByDistance: sort_key ascending by float_compare_to (NaN first).
const int SORT_KEY_FLOAT = 5;

int float_compare_to(float l, float r) {
    if (l < r) return -1;
    if (l > r) return 1;
    if (l == r) return 0;
    if (is_nan32(l)) return is_nan32(r) ? 0 : -1;
    return 1;
}

#ifndef SORT_CAP
#define SORT_CAP MAX_HORSES
#endif

int sort_mode;
int sort_h[SORT_CAP];
float sort_key[SORT_CAP];

int sort_cmp(int a, int b) {
    if (sort_mode == SORT_CUR_ORDER) {
        float xd = h_distance(uint(a)), yd = h_distance(uint(b));
        if (xd > yd) return -1;
        if (xd < yd) return 1;
        if (xd != yd) {
            if (!is_nan32(yd)) return 1;
            if (!is_nan32(xd)) return -1;
        }
        return sign(a - b);
    }
    if (sort_mode == SORT_FINISH_TIME) {
        double x = h_finish_time_for_order(uint(a)), y = h_finish_time_for_order(uint(b));
        if (x < y) return -1;
        if (x > y) return 1;
        bool xn = is_nan64(x), yn = is_nan64(y);
        if (xn && !yn) return -1;
        if (!xn && yn) return 1;
        return sign(a - b);
    }
    if (sort_mode == SORT_ORDER) return h_cur_order(uint(a)) - h_cur_order(uint(b));
    if (sort_mode == SORT_KEY_TRUNC) return f32_to_i32((sort_key[a] - sort_key[b]) * 10000.0);
    if (sort_mode == SORT_KEY_FLOAT) return float_compare_to(sort_key[a], sort_key[b]);
    return (f32_to_i32(h_lane_distance(uint(a))) - f32_to_i32(h_lane_distance(uint(b)))) * 10000;
}

void sort_swap(int i, int j) {
    int t = sort_h[i];
    sort_h[i] = sort_h[j];
    sort_h[j] = t;
}
void sort_swap_if_greater(int i, int j) {
    if (i != j && sort_cmp(sort_h[i], sort_h[j]) > 0) sort_swap(i, j);
}

int sort_partition(int lo, int hi) {
    int mid = lo + (hi - lo) / 2;
    sort_swap_if_greater(lo, mid);
    sort_swap_if_greater(lo, hi);
    sort_swap_if_greater(mid, hi);
    int pivot = sort_h[mid];
    if (mid != hi - 1) sort_swap(mid, hi - 1);
    int left = lo, right = hi - 1;
    while (left < right) {
        do { left++; } while (sort_cmp(sort_h[left], pivot) < 0);
        do { right--; } while (sort_cmp(pivot, sort_h[right]) < 0);
        if (left >= right) break;
        sort_swap(left, right);
    }
    if (left != hi - 1) sort_swap(left, hi - 1);
    return left;
}

void sort_insertion(int lo, int hi) {
    for (int i = lo; i < hi; i++) {
        int t = sort_h[i + 1];
        int j = i;
        while (j >= lo && sort_cmp(t, sort_h[j]) < 0) {
            sort_h[j + 1] = sort_h[j];
            j--;
        }
        sort_h[j + 1] = t;
    }
}

// Sorts sort_h[0..length). The recursion is a stack, right part first like the recursive call (the ranges are
// disjoint, so the order is immaterial anyway). Returns false on the heapsort fallback, which is unreachable for
// n <= 18 and not ported (the depth limit is 2 * floor(log2 n) + 2, and the stack holds at most that many ranges).
bool intro_sort(int length) {
    if (length < 2) return true;
    int depth = 0;
    int n = length;
    while (true) {
        n >>= 1;
        depth++;
        if (n < 1) break;
    }
    int stack_lo[32], stack_hi[32], stack_depth[32];
    stack_lo[0] = 0;
    stack_hi[0] = length - 1;
    stack_depth[0] = 2 * depth;
    int top = 1;
    while (top > 0) {
        top--;
        int lo = stack_lo[top], hi = stack_hi[top], depth_limit = stack_depth[top];
        while (hi > lo) {
            int size = hi - lo + 1;
            if (size <= 16) {
                if (size == 2) {
                    sort_swap_if_greater(lo, hi);
                } else if (size == 3) {
                    sort_swap_if_greater(lo, hi - 1);
                    sort_swap_if_greater(lo, hi);
                    sort_swap_if_greater(hi - 1, hi);
                } else if (size > 3) {
                    sort_insertion(lo, hi);
                }
                break;
            }
            if (depth_limit == 0) return false;
            depth_limit--;
            int p = sort_partition(lo, hi);
            stack_lo[top] = lo;
            stack_hi[top] = p - 1;
            stack_depth[top] = depth_limit;
            top++;
            lo = p + 1;
        }
    }
    return true;
}

#endif
