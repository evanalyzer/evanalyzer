//! # tls
//!
//! TLS for client connections (`wss://`), configured in `[tls]` (see
//! [`TlsConfig`]).
//!
//! Without a configured certificate the server makes its own self-signed one
//! on first start and keeps it, so its fingerprint - which clients pin with
//! `--remote-fingerprint` - stays the same across restarts. No CA, no
//! certificate files to manage: encrypted out of the box.
//!
//! Only the client side is encrypted; workers listen on 127.0.0.1 and are
//! reached in plain text.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::config::TlsConfig;
use log::info;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};
use std::{fs, io, path::Path, sync::Arc};

const CERT_FILE: &str = "cert.pem";
const KEY_FILE: &str = "key.pem";

/// The TLS setup clients are accepted with.
pub struct Tls {
    pub config: Arc<rustls::ServerConfig>,
    /// SHA-256 of the server certificate, `AB:CD:...` - what clients pass
    /// as `--remote-fingerprint`.
    pub fingerprint: String,
}

/// TLS as configured, `None` if `tls.enabled = false`. Loads `tls.cert` and
/// `tls.key`, or else the self-signed certificate in `self_signed_dir`
/// (see [`crate::ServerConfig::self_signed_dir`]), creating it on first
/// start.
pub fn load(config: &TlsConfig, self_signed_dir: &Path) -> io::Result<Option<Tls>> {
    if !config.enabled {
        return Ok(None);
    }
    let (cert, key) = match (&config.cert, &config.key) {
        (Some(cert), Some(key)) => (cert.clone(), key.clone()),
        // Both or neither - checked with the rest of the config.
        _ => {
            ensure_self_signed(self_signed_dir)?;
            (
                self_signed_dir.join(CERT_FILE),
                self_signed_dir.join(KEY_FILE),
            )
        }
    };
    let chain: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(&cert)
        .and_then(|certs| certs.collect())
        .map_err(|e| pem_error(&cert, e))?;
    let Some(leaf) = chain.first() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} holds no certificate", cert.display()),
        ));
    };
    let fingerprint = fingerprint(leaf);
    let key = PrivateKeyDer::from_pem_file(&key).map_err(|e| pem_error(&key, e))?;
    let mut server_config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(io::Error::other)?
    .with_no_client_auth()
    .with_single_cert(chain, key)
    .map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("TLS certificate {}: {e}", cert.display()),
        )
    })?;
    // Clients connect once and stay: resumption would only add state.
    server_config.send_tls13_tickets = 0;
    info!(
        "TLS certificate {}, fingerprint {fingerprint}",
        cert.display()
    );
    Ok(Some(Tls {
        config: Arc::new(server_config),
        fingerprint,
    }))
}

/// SHA-256 of a DER certificate as `AB:CD:...`.
pub fn fingerprint(cert: &[u8]) -> String {
    ring::digest::digest(&ring::digest::SHA256, cert)
        .as_ref()
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(":")
}

/// Creates a self-signed certificate and its key in `dir`, unless both are
/// there already. The key is readable by the server's account only.
fn ensure_self_signed(dir: &Path) -> io::Result<()> {
    let (cert_path, key_path) = (dir.join(CERT_FILE), dir.join(KEY_FILE));
    if cert_path.exists() && key_path.exists() {
        return Ok(());
    }
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(dir).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!(
                "Cannot create {} for the TLS certificate: {e}",
                dir.display()
            ),
        )
    })?;
    // Clients pin the fingerprint, not the name - but a name is required.
    let generated = rcgen::generate_simple_self_signed(vec!["evanalyzer-server".to_string()])
        .map_err(io::Error::other)?;
    write_private(&key_path, generated.signing_key.serialize_pem().as_bytes())?;
    fs::write(&cert_path, generated.cert.pem())?;
    info!(
        "Created a self-signed TLS certificate in {} - clients pin its fingerprint",
        dir.display()
    );
    Ok(())
}

/// Writes a new file only the owner can read.
fn write_private(path: &Path, contents: &[u8]) -> io::Result<()> {
    use std::io::Write;
    let _ = fs::remove_file(path);
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    options.open(path)?.write_all(contents)
}

fn pem_error(path: &Path, e: rustls::pki_types::pem::Error) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("Cannot read {}: {e}", path.display()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_tls_loads_nothing() {
        let config = TlsConfig {
            enabled: false,
            ..TlsConfig::default()
        };
        let dir = tempfile::tempdir().unwrap();
        assert!(load(&config, dir.path()).unwrap().is_none());
        assert!(
            fs::read_dir(dir.path()).unwrap().next().is_none(),
            "nothing generated"
        );
    }

    #[test]
    fn the_self_signed_certificate_is_created_once_and_kept() {
        let dir = tempfile::tempdir().unwrap();
        let tls_dir = dir.path().join("tls");

        let first = load(&TlsConfig::default(), &tls_dir).unwrap().unwrap();
        let again = load(&TlsConfig::default(), &tls_dir).unwrap().unwrap();

        assert_eq!(first.fingerprint, again.fingerprint, "same after a restart");
        assert_eq!(first.fingerprint.len(), 32 * 3 - 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&tls_dir.join(KEY_FILE)), 0o600);
            assert_eq!(mode(&tls_dir), 0o700);
        }
    }

    #[test]
    fn a_configured_certificate_is_used_as_is() {
        let dir = tempfile::tempdir().unwrap();
        let generated = rcgen::generate_simple_self_signed(vec!["server.lab".to_string()]).unwrap();
        let (cert, key) = (dir.path().join("c.pem"), dir.path().join("k.pem"));
        fs::write(&cert, generated.cert.pem()).unwrap();
        fs::write(&key, generated.signing_key.serialize_pem()).unwrap();

        let tls = load(
            &TlsConfig {
                cert: Some(cert),
                key: Some(key),
                ..TlsConfig::default()
            },
            &dir.path().join("unused"),
        )
        .unwrap()
        .unwrap();

        assert_eq!(tls.fingerprint, fingerprint(generated.cert.der()));
        assert!(!dir.path().join("unused").exists(), "nothing generated");
    }

    #[test]
    fn unreadable_certificate_files_are_reported_with_their_path() {
        let dir = tempfile::tempdir().unwrap();
        let cert = dir.path().join("cert.pem");
        fs::write(&cert, "not a certificate").unwrap();
        let error = load(
            &TlsConfig {
                cert: Some(cert),
                key: Some(dir.path().join("missing.pem")),
                ..TlsConfig::default()
            },
            &dir.path().join("unused"),
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("cert.pem"), "{error}");
    }
}
