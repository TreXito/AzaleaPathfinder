use std::sync::OnceLock;

pub(crate) fn navigation_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PF_NAV_DEBUG").is_some())
}

pub(crate) fn parkour_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PF_PARKOUR_DEBUG").is_some())
}
