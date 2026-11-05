//! A bump arena behind the global allocator. The allocations of one race come from one contiguous region and are all
//! released at once, which spares the system heap thousands of calls per race and keeps the race state close together.
//!
//! A final artifact opts in by declaring [`ArenaAllocator`]. Without that declaration every scope has no effect.
//! Each worker thread then holds a [`ThreadArena`] and opens one [`ArenaScope`] per race:
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOCATOR: arena::ArenaAllocator = arena::ArenaAllocator;
//!
//! let _arena = arena::ThreadArena::attach();
//! for input in &inputs {
//!     // SAFETY: The output ends with the scope, and the summaries hold nothing of it.
//!     let _scope = unsafe { arena::open_scope() };
//!     let output = simulate(input, data);
//!     arena::outside(|| summaries.push(summarize(&output)));
//! }
//! ```
//!
//! Debug builds check the contract of [`open_scope`]: they count the allocations of each scope and overwrite its memory
//! when it ends.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::marker::PhantomData;
use std::num::NonZero;
use std::ptr;
use std::sync::Once;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::{Acquire, Relaxed, Release}};
use std::thread;

/// Bytes of one region, twice what the largest race needs (1.8 MiB, a runner watching 400 skills). A race that outgrows
/// its region continues on the system heap.
const REGION_BYTES: usize = 4 << 20;

/// Upper limit of the region count: [`FREE_REGIONS`] holds one bit per region.
const MAX_REGIONS: usize = 64;

/// Alignment of the reservation, which starts each region on a page.
const RESERVATION_ALIGNMENT: usize = 4096;

/// Written over the memory of an ended scope in debug builds, so that a dangling read fails visibly.
const RELEASED_MEMORY_PATTERN: u8 = 0xDD;

static RESERVATION_ONCE: Once = Once::new();
static RESERVATION_BASE: AtomicUsize = AtomicUsize::new(0);
static RESERVATION_BYTES: AtomicUsize = AtomicUsize::new(0);
static FREE_REGIONS: AtomicU64 = AtomicU64::new(0);

struct ThreadState {
    /// Start of the attached region, or 0 without one.
    region: Cell<usize>,
    cursor: Cell<usize>,
    /// End of the region while a scope is open, otherwise 0, which sends every new allocation to the system heap.
    open_limit: Cell<usize>,
    /// Allocations of the open scope that are not yet released. Debug builds only.
    live_allocations: Cell<usize>,
}

thread_local! {
    static STATE: ThreadState = const {
        ThreadState {
            region: Cell::new(0),
            cursor: Cell::new(0),
            open_limit: Cell::new(0),
            live_allocations: Cell::new(0),
        }
    };
}

/// Reserves the regions on the system heap, once per process. An untouched page takes no physical memory. A failed
/// reservation leaves no regions, and every race then runs on the system heap.
fn reserve_regions() {
    let count = thread::available_parallelism().map_or(1, NonZero::get).min(MAX_REGIONS);
    let Ok(layout) = Layout::from_size_align(count * REGION_BYTES, RESERVATION_ALIGNMENT) else { return };
    let base = unsafe { System.alloc(layout) };
    if base.is_null() {
        return;
    }
    RESERVATION_BASE.store(base.expose_provenance(), Relaxed);
    RESERVATION_BYTES.store(layout.size(), Release);
    FREE_REGIONS.store(u64::MAX >> (MAX_REGIONS - count), Relaxed);
}

#[inline]
fn is_region_pointer(pointer: *mut u8) -> bool {
    let bytes = RESERVATION_BYTES.load(Acquire);
    pointer.addr().wrapping_sub(RESERVATION_BASE.load(Relaxed)) < bytes
}

/// Returns memory from the open scope, or null when no scope is open, the region cannot hold the layout or the thread
/// is panicking. A panic payload crosses scopes and threads, and the backtrace cache of the standard library lasts for
/// the process, so what a panic allocates stays on the system heap.
#[inline]
fn bump(layout: Layout) -> *mut u8 {
    STATE.with(|state| {
        let start = (state.cursor.get() + layout.align() - 1) & !(layout.align() - 1);
        let end = start + layout.size();
        if end > state.open_limit.get() || thread::panicking() {
            return ptr::null_mut();
        }
        state.cursor.set(end);
        if cfg!(debug_assertions) {
            state.live_allocations.set(state.live_allocations.get() + 1);
        }
        ptr::with_exposed_provenance_mut(start)
    })
}

#[inline]
fn note_release() {
    if cfg!(debug_assertions) {
        STATE.with(|state| state.live_allocations.set(state.live_allocations.get().saturating_sub(1)));
    }
}

/// Global allocator that serves the open [`ArenaScope`] of the calling thread and otherwise the system heap. Region
/// memory may be released from any thread, which does nothing.
pub struct ArenaAllocator;

unsafe impl GlobalAlloc for ArenaAllocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = bump(layout);
        if pointer.is_null() {
            return unsafe { System.alloc(layout) };
        }
        pointer
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = bump(layout);
        if pointer.is_null() {
            return unsafe { System.alloc_zeroed(layout) };
        }
        unsafe { pointer.write_bytes(0, layout.size()) };
        pointer
    }

    #[inline]
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if is_region_pointer(pointer) {
            return note_release();
        }
        unsafe { System.dealloc(pointer, layout) }
    }

    #[inline]
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if !is_region_pointer(pointer) {
            return unsafe { System.realloc(pointer, layout, new_size) };
        }
        let extended = STATE.with(|state| {
            let address = pointer.addr();
            let is_last = address + layout.size() == state.cursor.get();
            let fits = is_last && address + new_size <= state.open_limit.get();
            if fits {
                state.cursor.set(address + new_size);
            }
            fits
        });
        if extended || new_size <= layout.size() {
            return pointer;
        }
        unsafe {
            let moved = self.alloc(Layout::from_size_align_unchecked(new_size, layout.align()));
            if !moved.is_null() {
                ptr::copy_nonoverlapping(pointer, moved, layout.size());
                note_release();
            }
            moved
        }
    }
}

/// The region of the calling thread, which returns to the pool when this value drops. Regions keep their touched pages,
/// so later threads start warm. A thread beyond the region count runs on the system heap. The first attachment in a
/// process reserves the regions. Must not be attached inside a scope or while another `ThreadArena` of this thread is
/// alive.
#[must_use]
pub struct ThreadArena {
    thread_bound: PhantomData<*mut ()>,
}

impl ThreadArena {
    /// Attaches a free region to the calling thread.
    ///
    /// # Returns
    /// The owner of the attachment. No error is possible: without a free region the thread uses the system heap.
    pub fn attach() -> ThreadArena {
        RESERVATION_ONCE.call_once(reserve_regions);
        let taken = FREE_REGIONS.fetch_update(Acquire, Relaxed, |free| (free != 0).then(|| free & (free - 1)));
        let region = taken.map_or(0, |free| RESERVATION_BASE.load(Relaxed) + free.trailing_zeros() as usize * REGION_BYTES);
        STATE.with(|state| {
            debug_assert_eq!(state.region.get(), 0, "a thread holds one region at a time");
            state.region.set(region);
        });
        ThreadArena { thread_bound: PhantomData }
    }
}

impl Drop for ThreadArena {
    fn drop(&mut self) {
        let region = STATE.with(|state| state.region.replace(0));
        if region != 0 {
            let slot = (region - RESERVATION_BASE.load(Relaxed)) / REGION_BYTES;
            FREE_REGIONS.fetch_or(1 << slot, Release);
        }
    }
}

/// An open arena on the calling thread, which ends when this value drops. Declare it before the race, so that the race
/// drops first.
#[must_use]
pub struct ArenaScope {
    thread_bound: PhantomData<*mut ()>,
}

/// Sends the allocations of the calling thread to its region until the returned value drops. Without a region, or
/// without [`ArenaAllocator`] as the global allocator, this has no effect.
///
/// # Returns
/// The scope, which owns all memory allocated through it: dropping it makes that memory reusable.
///
/// # Safety
/// Nothing allocated inside the scope may be used after the scope ends. Allocate anything that lasts longer inside
/// [`outside`], or before the scope. Scopes do not nest.
pub unsafe fn open_scope() -> ArenaScope {
    STATE.with(|state| {
        debug_assert_eq!(state.open_limit.get(), 0, "arena scopes do not nest");
        let region = state.region.get();
        state.cursor.set(region);
        if region != 0 {
            state.open_limit.set(region + REGION_BYTES);
        }
    });
    ArenaScope { thread_bound: PhantomData }
}

impl Drop for ArenaScope {
    fn drop(&mut self) {
        STATE.with(|state| {
            state.open_limit.set(0);
            if cfg!(debug_assertions) {
                let region = state.region.get();
                if region != 0 {
                    unsafe { ptr::with_exposed_provenance_mut::<u8>(region).write_bytes(RELEASED_MEMORY_PATTERN, state.cursor.get() - region) };
                }
                let live_allocations = state.live_allocations.replace(0);
                debug_assert!(live_allocations == 0 || thread::panicking(), "{live_allocations} allocations outlive their arena scope");
            }
        });
    }
}

/// Runs `body` with the allocations of the calling thread on the system heap, for values that outlive the open scope.
///
/// # Returns
/// The value of `body`.
pub fn outside<T>(body: impl FnOnce() -> T) -> T {
    let open_limit = STATE.with(|state| state.open_limit.replace(0));
    let value = body();
    STATE.with(|state| state.open_limit.set(open_limit));
    value
}
