//! The race's random number generator: .NET Framework `System.Random` and the race wrapper around it.

const MBIG: i32 = i32::MAX;
const MSEED: i32 = 161803398;

/// .NET Framework / Mono `System.Random` (Knuth subtractive generator).
#[derive(Clone, Debug)]
pub struct NetRandom {
    seed_array: [i32; 56],
    inext: usize,
    inextp: usize,
    /// Number of `internal_sample` calls so far.
    pub draws: u64,
}

impl NetRandom {
    pub fn new(seed: i32) -> Self {
        let mut sa = [0i32; 56];
        let subtraction = if seed == i32::MIN { MBIG } else { seed.abs() };
        let mut mj = MSEED - subtraction;
        sa[55] = mj;
        let mut mk = 1i32;
        let mut ii = 0usize;
        for _ in 1..55 {
            ii += 21;
            if ii >= 55 {
                ii -= 55;
            }
            sa[ii] = mk;
            mk = mj.wrapping_sub(mk);
            if mk < 0 {
                mk = mk.wrapping_add(MBIG);
            }
            mj = sa[ii];
        }
        for _ in 1..5 {
            for i in 1..56 {
                let mut n = i + 30;
                if n >= 55 {
                    n -= 55;
                }
                sa[i] = sa[i].wrapping_sub(sa[1 + n]);
                if sa[i] < 0 {
                    sa[i] = sa[i].wrapping_add(MBIG);
                }
            }
        }
        NetRandom { seed_array: sa, inext: 0, inextp: 21, draws: 0 }
    }

    pub fn internal_sample(&mut self) -> i32 {
        let mut a = self.inext + 1;
        if a >= 56 {
            a = 1;
        }
        let mut b = self.inextp + 1;
        if b >= 56 {
            b = 1;
        }
        let mut r = self.seed_array[a].wrapping_sub(self.seed_array[b]);
        if r == MBIG {
            r -= 1;
        }
        if r < 0 {
            r = r.wrapping_add(MBIG);
        }
        self.seed_array[a] = r;
        self.inext = a;
        self.inextp = b;
        self.draws += 1;
        r
    }

    /// The whole state as 60 words: the seed array, `inext`, `inextp`, then `draws` (low word first). The GPU keeps
    /// the generator in this form.
    pub fn words(&self) -> [u32; 60] {
        let mut w = [0u32; 60];
        for (k, &v) in self.seed_array.iter().enumerate() {
            w[k] = v as u32;
        }
        w[56] = self.inext as u32;
        w[57] = self.inextp as u32;
        w[58] = self.draws as u32;
        w[59] = (self.draws >> 32) as u32;
        w
    }

    pub fn next_double(&mut self) -> f64 {
        self.internal_sample() as f64 * (1.0 / MBIG as f64)
    }

    /// `Next(min, max)`, max exclusive.
    pub fn next_range(&mut self, min: i32, max: i32) -> i32 {
        assert!(min <= max, "Random.Next: minValue > maxValue");
        let range = max as i64 - min as i64;
        if range <= i32::MAX as i64 {
            // Truncation; the product is always in i32 range here.
            (self.next_double() * range as f64) as i32 + min
        } else {
            // GetSampleForLargeRange: never hit by race code.
            let r = self.internal_sample();
            let neg = self.internal_sample() % 2 == 0;
            let r = if neg { -r } else { r } as f64;
            let d = (r + (MBIG as f64 - 1.0)) / (2.0 * MBIG as f64 - 1.0);
            ((d * range as f64) as i64 + min as i64) as i32
        }
    }
}

#[derive(Clone, Debug)]
pub struct RaceRandom {
    pub random: NetRandom,
}

impl RaceRandom {
    pub fn new(seed: i32) -> Self {
        RaceRandom { random: NetRandom::new(seed) }
    }

    /// Number of `internal_sample` calls so far.
    pub fn draws(&self) -> u64 {
        self.random.draws
    }

    /// `Next(min, max)`, max exclusive.
    pub fn random_int_range(&mut self, min: i32, max: i32) -> i32 {
        self.random.next_range(min, max)
    }

    /// `Next(0, max)` (not `Next(max)`).
    pub fn random_int(&mut self, max: i32) -> i32 {
        self.random.next_range(0, max)
    }

    /// `(float)NextDouble() * |max - min| + min`, clamped to [min, max] (NaN passes through).
    pub fn random_float_range(&mut self, min: f32, max: f32) -> f32 {
        let d = self.random.next_double();
        // The product and sum are evaluated in f64 with one final rounding.
        let f = (d as f32 as f64 * (max - min).abs() as f64 + min as f64) as f32;
        if min > f {
            min
        } else if f > max {
            max
        } else {
            f
        }
    }

    /// `random_float_range(0, max)`.
    pub fn random_float(&mut self, max: f32) -> f32 {
        self.random_float_range(0.0, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values from the Python reference `NetRandom`.
    #[test]
    fn matches_python_reference() {
        for &(seed, ref expected) in EXPECTED {
            let mut r = NetRandom::new(seed);
            for (i, &e) in expected.iter().enumerate() {
                assert_eq!(r.internal_sample(), e, "seed {seed} draw {i}");
            }
        }
    }

    #[test]
    fn overloads() {
        // GetRandom(float max) = (float)d * max, and draw counting.
        let mut a = NetRandom::new(12345);
        let mut b = RaceRandom::new(12345);
        let d = a.next_double();
        assert_eq!(b.random_float(0.1), d as f32 * 0.1f32);
        let d = a.next_double();
        assert_eq!(b.random_int(7), (d * 7.0) as i32);
        let d = a.next_double();
        assert_eq!(b.random_int_range(-3, 5), (d * 8.0) as i32 - 3);
        assert_eq!(b.draws(), 3);
    }

    #[rustfmt::skip]
    const EXPECTED: &[(i32, [i32; 60])] = &[
        (0, [1559595546, 1755192844, 1649316166, 1198642031, 442452829, 1200195957, 1945678308, 949569752, 2099272109, 587775847, 626863973, 1003550677, 1358625013, 1008269081, 2109153755, 65212616, 1851925803, 2137491580, 1454235444, 675580731, 1754296375, 1821177336, 2130093701, 70062080, 1503113964, 1130186590, 2005789796, 1476653312, 1174277203, 174182291, 401846963, 973512717, 638171722, 2122881600, 1380182313, 1638451829, 65271247, 818200948, 736891500, 2056119311, 1084756724, 1537539262, 255459778, 587232589, 1947978014, 1706746116, 724046315, 981848395, 315304373, 475269784, 880625662, 1543454120, 1331075398, 1047903413, 418573418, 1885901857, 1772582790, 1579254086, 1843011714, 1459749886]),
        (1, [534011718, 237820880, 1002897798, 1657007234, 1412011072, 929393559, 760389092, 2026928803, 217468053, 1379662799, 61497087, 532638534, 687431273, 2125508764, 1464848243, 1406361028, 607156385, 1321446349, 1512368656, 1507096884, 2039633683, 200550235, 345545550, 820276643, 1713579631, 363927940, 1704637349, 659767626, 1767440095, 1903678842, 1194650126, 1539174560, 1501726364, 29236707, 2130037697, 1731519511, 1844137836, 134224868, 1104652174, 1127983490, 585887566, 2134161410, 1484135457, 830250430, 932725007, 1450422331, 1454666067, 1015490648, 525100016, 759101347, 1130118952, 363842328, 744291659, 1485403625, 848563345, 333461483, 2039758977, 182621155, 2090911250, 1048083132]),
        (-1, [534011718, 237820880, 1002897798, 1657007234, 1412011072, 929393559, 760389092, 2026928803, 217468053, 1379662799, 61497087, 532638534, 687431273, 2125508764, 1464848243, 1406361028, 607156385, 1321446349, 1512368656, 1507096884, 2039633683, 200550235, 345545550, 820276643, 1713579631, 363927940, 1704637349, 659767626, 1767440095, 1903678842, 1194650126, 1539174560, 1501726364, 29236707, 2130037697, 1731519511, 1844137836, 134224868, 1104652174, 1127983490, 585887566, 2134161410, 1484135457, 830250430, 932725007, 1450422331, 1454666067, 1015490648, 525100016, 759101347, 1130118952, 363842328, 744291659, 1485403625, 848563345, 333461483, 2039758977, 182621155, 2090911250, 1048083132]),
        (-2147483648, [1559595546, 1755192844, 1649316172, 1198642031, 442452829, 1200195955, 1945678308, 949569752, 2099272109, 587775835, 626863973, 1003550677, 1358624999, 1008269081, 2109153755, 65212616, 1851925783, 2137491580, 1454235444, 675580701, 1754296375, 1821177336, 2130093701, 70062072, 1503113964, 1130186590, 2005789778, 1476653312, 1174277203, 174182293, 401846963, 973512717, 638171722, 2122881612, 1380182313, 1638451829, 65271241, 818200948, 736891500, 2056119313, 1084756754, 1537539262, 255459778, 587232601, 1947978014, 1706746116, 724046329, 981848415, 315304373, 475269784, 880625680, 1543454120, 1331075398, 1047903443, 418573420, 1885901857, 1772582790, 1579254100, 1843011714, 1459749886]),
        (2147483647, [1559595546, 1755192844, 1649316172, 1198642031, 442452829, 1200195955, 1945678308, 949569752, 2099272109, 587775835, 626863973, 1003550677, 1358624999, 1008269081, 2109153755, 65212616, 1851925783, 2137491580, 1454235444, 675580701, 1754296375, 1821177336, 2130093701, 70062072, 1503113964, 1130186590, 2005789778, 1476653312, 1174277203, 174182293, 401846963, 973512717, 638171722, 2122881612, 1380182313, 1638451829, 65271241, 818200948, 736891500, 2056119313, 1084756754, 1537539262, 255459778, 587232601, 1947978014, 1706746116, 724046329, 981848415, 315304373, 475269784, 880625680, 1543454120, 1331075398, 1047903443, 418573420, 1885901857, 1772582790, 1579254100, 1843011714, 1459749886]),
        (12345, [143337951, 150666398, 1663795458, 1097663221, 1712597933, 1776631026, 356393799, 1580828476, 558810388, 1086637143, 494509053, 831377771, 463814839, 44691035, 289552956, 1590924033, 418954878, 1904902962, 1849199486, 770656856, 222698908, 1137270943, 770420532, 1519356451, 1246560209, 1332617375, 1573024538, 1606065954, 850942673, 526685912, 1473914819, 452144508, 1111403504, 1042369160, 542895576, 1655234974, 5538230, 1039193352, 961982272, 1044665811, 1528100810, 969047112, 579718272, 607824875, 1364170491, 633032322, 793567355, 1831117809, 377238926, 1830086762, 1383740914, 1322492187, 948158774, 1066648348, 64646849, 1153550655, 1527729513, 144439007, 1998586659, 379980558]),
        (1857374301, [1058686674, 1211286027, 2123253641, 1197639276, 1104563502, 840456688, 854107627, 951638204, 1999624660, 1073599023, 1401382231, 780589672, 1163422296, 1184622415, 1715585905, 522559867, 2014438014, 1510109197, 1283843776, 241625248, 1869670102, 2086706863, 2076353736, 973854009, 557194513, 4761722, 1235953175, 1553866626, 1976550455, 1988548862, 7102276, 1197508346, 820099477, 1716487362, 1119757803, 219802518, 1199122219, 11624638, 74600309, 363655930, 1731372103, 1901921978, 1207741779, 65479913, 555574922, 1775357636, 1110329820, 2037061358, 111521021, 901781776, 714051180, 1548371148, 638798927, 604538498, 1296324237, 1119463458, 1282415938, 1149399632, 640444763, 1099801780]),
    ];
}
