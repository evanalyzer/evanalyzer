//! # login_throttle
//!
//! Slows down password guessing: after a few failed logins from one address
//! its next tries are refused - without checking the password, which costs
//! the server a deliberately slow hash each time - for a time that doubles
//! with each further failure.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-08
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use std::{
    collections::HashMap,
    net::{IpAddr, Ipv6Addr},
    sync::Mutex,
    time::{Duration, Instant},
};

/// Failed logins an address may have before it has to wait.
const FREE_TRIES: u32 = 5;
/// The longest wait: 1 s after the first failure beyond the free ones, then
/// doubling up to this.
const MAX_WAIT: Duration = Duration::from_secs(5 * 60);
/// An address without a failure for this long starts over.
const FORGET_AFTER: Duration = Duration::from_secs(15 * 60);
/// Addresses tracked before old ones are forgotten early - bounds the
/// memory an attacker with many addresses can make the server use.
const MAX_TRACKED: usize = 10_000;

struct Failures {
    count: u32,
    last: Instant,
}

impl Failures {
    /// When the next try is allowed again.
    fn allowed_from(&self) -> Instant {
        if self.count < FREE_TRIES {
            return self.last;
        }
        let doublings = (self.count - FREE_TRIES).min(16);
        self.last + Duration::from_secs(1 << doublings).min(MAX_WAIT)
    }
}

/// Failed logins per client address, shared by all connections.
#[derive(Default)]
pub(crate) struct LoginThrottle {
    failures: Mutex<HashMap<IpAddr, Failures>>,
}

impl LoginThrottle {
    /// How long `ip` still has to wait before it may try to log in again;
    /// `None`: it may now.
    pub fn wait(&self, ip: IpAddr, now: Instant) -> Option<Duration> {
        let failures = self.failures.lock().unwrap();
        let entry = failures.get(&key(ip))?;
        let wait = entry.allowed_from().saturating_duration_since(now);
        (!wait.is_zero()).then_some(wait)
    }

    /// A login (or resume) from `ip` failed.
    pub fn failed(&self, ip: IpAddr, now: Instant) {
        let mut failures = self.failures.lock().unwrap();
        if failures.len() >= MAX_TRACKED {
            failures.retain(|_, f| now.saturating_duration_since(f.last) < FORGET_AFTER);
            if failures.len() >= MAX_TRACKED {
                // Still full: keep the addresses that already have to wait.
                failures.retain(|_, f| f.count >= FREE_TRIES);
            }
        }
        let entry = failures.entry(key(ip)).or_insert(Failures {
            count: 0,
            last: now,
        });
        if now.saturating_duration_since(entry.last) >= FORGET_AFTER {
            entry.count = 0;
        }
        entry.count = entry.count.saturating_add(1);
        entry.last = now;
    }

    /// A login from `ip` succeeded: its failures are forgotten.
    pub fn succeeded(&self, ip: IpAddr) {
        self.failures.lock().unwrap().remove(&key(ip));
    }
}

/// The address failures are counted for: an IPv4 address as is, an IPv6
/// address by its /64 network - one machine usually has a whole /64 and
/// could otherwise take a fresh address for each try.
fn key(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V4(_) => ip,
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => IpAddr::V4(v4),
            None => {
                let network = u128::from(v6) & !((1u128 << 64) - 1);
                IpAddr::V6(Ipv6Addr::from(network))
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(text: &str) -> IpAddr {
        text.parse().unwrap()
    }

    #[test]
    fn a_few_failures_are_free_then_the_wait_doubles_up_to_five_minutes() {
        let throttle = LoginThrottle::default();
        let (alice, t0) = (ip("192.0.2.1"), Instant::now());
        for _ in 0..FREE_TRIES - 1 {
            throttle.failed(alice, t0);
            assert_eq!(throttle.wait(alice, t0), None);
        }
        throttle.failed(alice, t0);
        assert_eq!(throttle.wait(alice, t0), Some(Duration::from_secs(1)));
        assert_eq!(throttle.wait(alice, t0 + Duration::from_secs(1)), None);
        throttle.failed(alice, t0);
        assert_eq!(throttle.wait(alice, t0), Some(Duration::from_secs(2)));
        for _ in 0..30 {
            throttle.failed(alice, t0);
        }
        assert_eq!(throttle.wait(alice, t0), Some(MAX_WAIT));
        // Others aren't affected.
        assert_eq!(throttle.wait(ip("192.0.2.2"), t0), None);
    }

    #[test]
    fn a_success_or_a_quiet_quarter_hour_starts_over() {
        let throttle = LoginThrottle::default();
        let (alice, t0) = (ip("192.0.2.1"), Instant::now());
        for _ in 0..FREE_TRIES {
            throttle.failed(alice, t0);
        }
        throttle.succeeded(alice);
        assert_eq!(throttle.wait(alice, t0), None);

        for _ in 0..FREE_TRIES + 3 {
            throttle.failed(alice, t0);
        }
        let later = t0 + FORGET_AFTER;
        assert_eq!(throttle.wait(alice, later), None);
        throttle.failed(alice, later);
        assert_eq!(throttle.wait(alice, later), None, "counted from 1 again");
    }

    /// An IPv6 client can't dodge the wait with another address of its
    /// own network.
    #[test]
    fn ipv6_addresses_count_per_network() {
        let throttle = LoginThrottle::default();
        let t0 = Instant::now();
        for n in 0..FREE_TRIES {
            throttle.failed(ip(&format!("2001:db8:1:2::{n}")), t0);
        }
        assert!(throttle.wait(ip("2001:db8:1:2:abcd::1"), t0).is_some());
        assert_eq!(throttle.wait(ip("2001:db8:1:3::1"), t0), None);
        // An IPv4 client over an IPv6 socket is its IPv4 address.
        for _ in 0..FREE_TRIES {
            throttle.failed(ip("::ffff:192.0.2.7"), t0);
        }
        assert!(throttle.wait(ip("192.0.2.7"), t0).is_some());
    }

    #[test]
    fn many_addresses_use_bounded_memory() {
        let throttle = LoginThrottle::default();
        let t0 = Instant::now();
        let blocked = ip("198.51.100.1");
        for _ in 0..FREE_TRIES {
            throttle.failed(blocked, t0);
        }
        for n in 0..(MAX_TRACKED as u32 + 500) {
            throttle.failed(IpAddr::V4((0x0a00_0000 + n).into()), t0);
        }
        assert!(throttle.failures.lock().unwrap().len() <= MAX_TRACKED);
        assert!(
            throttle.wait(blocked, t0).is_some(),
            "a blocked one is kept"
        );
    }
}
