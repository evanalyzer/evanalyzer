//! # recent
//!
//! The projects and servers the user opened last, newest first.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::settings::AppSettings;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// How many servers [`AppSettings::recent_servers`] keeps.
const MAX_RECENT_SERVERS: usize = 10;

/// Projects the start page lists at most.
const MAX_RECENT_PROJECTS: usize = 12;

/// A server the user connected to: what the connect dialog offers again.
/// Never the password.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecentServer {
    pub url: String,
    pub user: String,
    /// The certificate fingerprint the user confirmed for this server - a
    /// different one later means the certificate changed. `None` for
    /// `ws://` and for certificates signed by a public authority.
    #[serde(default)]
    pub fingerprint: Option<String>,
}

/// A project opened or saved: what the start page offers again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecentProject {
    /// The project file.
    pub path: PathBuf,
    pub name: String,
    #[serde(default)]
    pub images: u32,
    #[serde(default)]
    pub pipelines: u32,
    /// When it was last opened or saved (seconds since 1970, UTC).
    #[serde(default)]
    pub opened_at: i64,
}

impl AppSettings {
    /// Puts `project` first in [`Self::recent_projects`], replacing an
    /// older entry for the same file.
    pub fn remember_project(&mut self, project: RecentProject) {
        self.recent_projects
            .retain(|known| known.path != project.path);
        self.recent_projects.insert(0, project);
        self.recent_projects.truncate(MAX_RECENT_PROJECTS);
    }

    /// Takes `path` off [`Self::recent_projects`].
    pub fn forget_project(&mut self, path: &std::path::Path) {
        self.recent_projects.retain(|known| known.path != path);
    }

    /// Puts `server` first in [`Self::recent_servers`], replacing an older
    /// entry for the same address and user.
    pub fn remember_server(&mut self, server: RecentServer) {
        self.recent_servers
            .retain(|known| !(known.url == server.url && known.user == server.user));
        self.recent_servers.insert(0, server);
        self.recent_servers.truncate(MAX_RECENT_SERVERS);
    }

    /// The fingerprint confirmed for `url`, if any (by any user: it's the
    /// server's certificate, not the user's).
    pub fn known_fingerprint(&self, url: &str) -> Option<&str> {
        self.recent_servers
            .iter()
            .find(|known| known.url == url && known.fingerprint.is_some())
            .and_then(|known| known.fingerprint.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(path: &str) -> RecentProject {
        RecentProject {
            path: PathBuf::from(path),
            name: path.into(),
            images: 0,
            pipelines: 0,
            opened_at: 0,
        }
    }

    #[test]
    fn a_remembered_project_goes_first_once_and_the_list_stays_short() {
        let mut settings = AppSettings::default();
        settings.remember_project(project("/a.evaproj"));
        settings.remember_project(project("/b.evaproj"));
        settings.remember_project(project("/a.evaproj"));
        let paths: Vec<_> = settings
            .recent_projects
            .iter()
            .map(|p| p.path.clone())
            .collect();
        assert_eq!(
            paths,
            [PathBuf::from("/a.evaproj"), PathBuf::from("/b.evaproj")]
        );
        for i in 0..20 {
            settings.remember_project(project(&format!("/{i}.evaproj")));
        }
        assert_eq!(settings.recent_projects.len(), MAX_RECENT_PROJECTS);
        settings.forget_project(std::path::Path::new("/19.evaproj"));
        assert_eq!(
            settings.recent_projects[0].path,
            PathBuf::from("/18.evaproj")
        );
    }

    #[test]
    fn settings_saved_before_recent_projects_still_load() {
        let settings: AppSettings = serde_json::from_str("{}").expect("old settings");
        assert!(settings.recent_projects.is_empty());
        assert!(!settings.open_last_project);
    }

    fn server(url: &str, user: &str, fingerprint: Option<&str>) -> RecentServer {
        RecentServer {
            url: url.into(),
            user: user.into(),
            fingerprint: fingerprint.map(str::to_string),
        }
    }

    #[test]
    fn the_latest_server_comes_first_once_and_the_list_stays_short() {
        let mut settings = AppSettings::default();
        settings.remember_server(server("wss://a", "alice", Some("AA")));
        settings.remember_server(server("wss://b", "alice", None));
        settings.remember_server(server("wss://a", "alice", Some("AA")));
        assert_eq!(
            settings.recent_servers,
            [
                server("wss://a", "alice", Some("AA")),
                server("wss://b", "alice", None)
            ]
        );
        // Another user on the same server is another entry, but the
        // server's fingerprint is known for both.
        settings.remember_server(server("wss://a", "bob", None));
        assert_eq!(settings.recent_servers.len(), 3);
        assert_eq!(settings.known_fingerprint("wss://a"), Some("AA"));
        assert_eq!(settings.known_fingerprint("wss://b"), None);

        for i in 0..20 {
            settings.remember_server(server(&format!("wss://{i}"), "x", None));
        }
        assert_eq!(settings.recent_servers.len(), MAX_RECENT_SERVERS);
        assert_eq!(settings.recent_servers[0].url, "wss://19");
    }
}
