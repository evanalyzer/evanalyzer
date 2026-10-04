//! The client side of `wss://`: TLS to an `evanalyzer server`.
//!
//! Which server certificate is trusted is the user's choice, [`TlsTrust`]:
//! by default one a public certificate authority signed for the host name in
//! the URL (e.g. Let's Encrypt); with `--remote-fingerprint` exactly the one
//! with that SHA-256 fingerprint (the server's own self-signed certificate,
//! its default); with `--no-tls-verification` any.
//!
//! An untrusted certificate fails the connection with its fingerprint in the
//! message, so the user can compare it with the one the server logs at
//! startup and pass it.

use crate::api::ConnectionSecurity;
use crate::backend::remote::wire::conn::Socket;
use evanalyzer_cfg::core_types::InternalErrors;
use rustls::client::WebPkiServerVerifier;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{ClientConnection, DigitallySignedStruct, SignatureScheme, StreamOwned};
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

/// Which TLS certificate a client trusts as the server's (`wss://` only).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum TlsTrust {
    /// One a public certificate authority signed for the server's host name.
    #[default]
    PublicAuthorities,
    /// Exactly the certificate with this SHA-256 fingerprint (hex, `:`
    /// optional) - for the server's self-signed certificate, whose
    /// fingerprint it logs at startup.
    Fingerprint(String),
    /// Any certificate. Still encrypted, but anyone in between could pose as
    /// the server and read the password - for tests and trusted networks.
    NoVerification,
}

/// The connection to a server: `ws://` or `wss://`.
pub(crate) enum NetStream {
    Plain(TcpStream),
    Tls {
        stream: Box<StreamOwned<ClientConnection, TcpStream>>,
        /// Whether the server's certificate was checked.
        verified: bool,
    },
}

impl NetStream {
    pub(crate) fn security(&self) -> ConnectionSecurity {
        match self {
            Self::Plain(_) => ConnectionSecurity::Unencrypted,
            Self::Tls { verified: true, .. } => ConnectionSecurity::Encrypted,
            Self::Tls {
                verified: false, ..
            } => ConnectionSecurity::EncryptedUnverified,
        }
    }
}

impl Read for NetStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Plain(stream) => stream.read(buf),
            Self::Tls { stream, .. } => stream.read(buf),
        }
    }
}

impl Write for NetStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Plain(stream) => stream.write(buf),
            Self::Tls { stream, .. } => stream.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Plain(stream) => stream.flush(),
            Self::Tls { stream, .. } => stream.flush(),
        }
    }
}

impl Socket for NetStream {
    fn tcp(&self) -> &TcpStream {
        match self {
            Self::Plain(stream) => stream,
            Self::Tls { stream, .. } => &stream.sock,
        }
    }
}

/// SHA-256 of a DER certificate as `AB:CD:...` - the form the server logs.
pub(crate) fn fingerprint(cert: &[u8]) -> String {
    ring::digest::digest(&ring::digest::SHA256, cert)
        .as_ref()
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(":")
}

/// `fingerprint` as typed by a user - hex, with or without `:` or spaces,
/// any case, optionally prefixed `SHA256:` - in the canonical form.
pub(crate) fn parse_fingerprint(fingerprint: &str) -> Result<String, InternalErrors> {
    let trimmed = fingerprint.trim();
    let hex: String = trimmed
        .strip_prefix("SHA256:")
        .or_else(|| trimmed.strip_prefix("sha256:"))
        .unwrap_or(trimmed)
        .chars()
        .filter(|c| !matches!(c, ':' | ' '))
        .collect();
    if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(InternalErrors::InvalidArgument(format!(
            "'{fingerprint}' is not a SHA-256 certificate fingerprint (64 hex digits, \
             e.g. as `evanalyzer server` logs it at startup)"
        )));
    }
    let hex = hex.to_ascii_uppercase();
    Ok((0..32)
        .map(|i| &hex[2 * i..2 * i + 2])
        .collect::<Vec<_>>()
        .join(":"))
}

/// Runs the TLS handshake with `host` on `stream`, accepting the server's
/// certificate as `trust` says. `url` is only for messages.
pub(crate) fn connect(
    stream: TcpStream,
    host: &str,
    trust: &TlsTrust,
    url: &str,
) -> Result<NetStream, InternalErrors> {
    let check = match trust {
        TlsTrust::PublicAuthorities => Check::PublicAuthorities,
        TlsTrust::Fingerprint(typed) => Check::Pinned(parse_fingerprint(typed)?),
        TlsTrust::NoVerification => Check::Nothing,
    };
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let verifier = Arc::new(Verifier {
        check,
        public_cas: WebPkiServerVerifier::builder_with_provider(
            Arc::new(rustls::RootCertStore {
                roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
            }),
            Arc::clone(&provider),
        )
        .build()
        .map_err(|e| InternalErrors::Internal(format!("TLS setup failed: {e}")))?,
        provider: Arc::clone(&provider),
        seen: Mutex::new(None),
    });
    let config = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| InternalErrors::Internal(format!("TLS setup failed: {e}")))?
        .dangerous()
        .with_custom_certificate_verifier(Arc::clone(&verifier) as Arc<dyn ServerCertVerifier>)
        .with_no_client_auth();
    // IPv6 hosts come bracketed out of the URL.
    let name = host.trim_start_matches('[').trim_end_matches(']');
    let name = ServerName::try_from(name.to_string())
        .map_err(|e| InternalErrors::InvalidArgument(format!("Invalid host in '{url}': {e}")))?;
    let mut tls = ClientConnection::new(Arc::new(config), name)
        .map_err(|e| InternalErrors::Internal(format!("TLS setup failed: {e}")))?;
    let mut stream = stream;
    // Handshake now rather than on first use, to report trust problems as such.
    while tls.is_handshaking() {
        if let Err(e) = tls.complete_io(&mut stream) {
            return Err(handshake_error(url, &verifier, e));
        }
    }
    let seen = verifier.seen.lock().unwrap().clone().unwrap_or_default();
    let verified = match &verifier.check {
        Check::PublicAuthorities => {
            log::info!("TLS: {url} has a certificate signed by a public authority");
            true
        }
        Check::Pinned(_) => {
            log::info!("TLS: {url} has the certificate with the given fingerprint");
            true
        }
        Check::Nothing => {
            log::warn!(
                "TLS: the certificate of {url} was NOT verified (--no-tls-verification): \
                 encrypted, but anyone in between could pose as the server. \
                 Its fingerprint is {seen}"
            );
            false
        }
    };
    Ok(NetStream::Tls {
        stream: Box::new(StreamOwned::new(tls, stream)),
        verified,
    })
}

/// Explains a failed handshake: an untrusted or unexpected certificate gets
/// its fingerprint and what to do; anything else is reported as is.
fn handshake_error(url: &str, verifier: &Verifier, error: io::Error) -> InternalErrors {
    let seen = verifier.seen.lock().unwrap().clone();
    let certificate_rejected = error
        .get_ref()
        .and_then(|e| e.downcast_ref::<rustls::Error>())
        .is_some_and(|e| matches!(e, rustls::Error::InvalidCertificate(_)));
    match (seen, &verifier.check) {
        (Some(seen), Check::Pinned(pinned)) if certificate_rejected => {
            InternalErrors::InvalidArgument(format!(
                "The certificate of {url} has the fingerprint\n  {seen}\nnot the expected\n  {pinned}\n\
             If the server's certificate was replaced on purpose, connect with the new \
             fingerprint. Otherwise someone may be intercepting the connection - don't log in."
            ))
        }
        (Some(seen), _) if certificate_rejected => InternalErrors::InvalidArgument(format!(
            "{url} has a certificate this computer can't verify ({error}) - normally the \
             server's own self-signed one. Its fingerprint is\n  {seen}\n\
             If that matches the fingerprint `evanalyzer server` logs at startup, connect with\n  \
             --remote-fingerprint {seen}"
        )),
        _ => InternalErrors::Io(format!(
            "TLS connection to {url} failed: {error} (if the server runs without TLS, \
             connect with ws:// instead)"
        )),
    }
}

/// What [`Verifier`] checks the certificate against - [`TlsTrust`] with the
/// fingerprint in canonical form.
#[derive(Debug)]
enum Check {
    PublicAuthorities,
    Pinned(String),
    Nothing,
}

/// Checks the server's certificate as [`Check`] says; remembers the
/// fingerprint it was shown, for messages.
#[derive(Debug)]
struct Verifier {
    check: Check,
    public_cas: Arc<WebPkiServerVerifier>,
    provider: Arc<CryptoProvider>,
    seen: Mutex<Option<String>>,
}

impl ServerCertVerifier for Verifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        let seen = fingerprint(end_entity);
        *self.seen.lock().unwrap() = Some(seen.clone());
        match &self.check {
            // The user vouched for exactly this certificate: name, issuer
            // and dates don't matter (a self-signed one has neither).
            Check::Pinned(pinned) if *pinned == seen => Ok(ServerCertVerified::assertion()),
            Check::Pinned(_) => Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::ApplicationVerificationFailure,
            )),
            Check::Nothing => Ok(ServerCertVerified::assertion()),
            Check::PublicAuthorities => self.public_cas.verify_server_cert(
                end_entity,
                intermediates,
                server_name,
                ocsp_response,
                now,
            ),
        }
    }

    // The handshake signatures are always checked: they prove the server
    // holds the certificate's private key (even unverified, the connection
    // is then encrypted for whoever holds it).
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// A TLS server with a fresh self-signed certificate for `names` that
    /// accepts one connection and echoes one line. Returns its address and
    /// certificate fingerprint.
    fn tls_server(names: &[&str]) -> (std::net::SocketAddr, String) {
        let generated = rcgen::generate_simple_self_signed(
            names.iter().map(|n| n.to_string()).collect::<Vec<_>>(),
        )
        .unwrap();
        let fingerprint = fingerprint(generated.cert.der());
        let config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(
            vec![generated.cert.der().clone()],
            rustls::pki_types::PrivateKeyDer::try_from(generated.signing_key.serialize_der())
                .unwrap(),
        )
        .unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let conn = rustls::ServerConnection::new(Arc::new(config)).unwrap();
            let mut tls = StreamOwned::new(conn, stream);
            let mut buf = [0u8; 5];
            if tls.read_exact(&mut buf).is_ok() {
                let _ = tls.write_all(&buf);
                let _ = tls.flush();
            }
        });
        (addr, fingerprint)
    }

    fn connect_to(addr: std::net::SocketAddr, trust: TlsTrust) -> Result<NetStream, String> {
        let stream = TcpStream::connect(addr).unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        connect(stream, "127.0.0.1", &trust, "wss://test").map_err(|e| e.to_string())
    }

    #[test]
    fn a_pinned_certificate_is_trusted_and_carries_data() {
        let (addr, fingerprint) = tls_server(&["evanalyzer-server"]);
        // As typed by a user: lower case, no colons, with prefix.
        let typed = format!("sha256:{}", fingerprint.replace(':', "").to_lowercase());

        let mut stream = connect_to(addr, TlsTrust::Fingerprint(typed)).unwrap();
        assert_eq!(stream.security(), ConnectionSecurity::Encrypted);

        stream.write_all(b"hello").unwrap();
        stream.flush().unwrap();
        let mut echo = [0u8; 5];
        stream.read_exact(&mut echo).unwrap();
        assert_eq!(&echo, b"hello");
    }

    #[test]
    fn an_unknown_self_signed_certificate_is_refused_with_its_fingerprint() {
        let (addr, fingerprint) = tls_server(&["127.0.0.1"]);
        let error = connect_to(addr, TlsTrust::PublicAuthorities).err().unwrap();
        assert!(error.contains(&fingerprint), "{error}");
        assert!(error.contains("--remote-fingerprint"), "{error}");
    }

    #[test]
    fn without_verification_any_certificate_is_accepted_but_marked_unverified() {
        let (addr, _) = tls_server(&["evanalyzer-server"]);
        let mut stream = connect_to(addr, TlsTrust::NoVerification).unwrap();
        assert_eq!(stream.security(), ConnectionSecurity::EncryptedUnverified);
        stream.write_all(b"hello").unwrap();
        stream.flush().unwrap();
        let mut echo = [0u8; 5];
        stream.read_exact(&mut echo).unwrap();
        assert_eq!(&echo, b"hello", "still a working, encrypted connection");
    }

    #[test]
    fn a_different_certificate_than_the_pinned_one_is_refused() {
        let (addr, fingerprint) = tls_server(&["evanalyzer-server"]);
        let other = "AA:".repeat(31) + "AA";
        let error = connect_to(addr, TlsTrust::Fingerprint(other))
            .err()
            .unwrap();
        assert!(error.contains(&fingerprint), "{error}");
        assert!(error.contains("intercepting"), "{error}");
    }

    #[test]
    fn fingerprints_are_accepted_in_the_usual_spellings_only() {
        let canonical = "AB:".repeat(31) + "CD";
        for typed in [
            canonical.clone(),
            canonical.to_lowercase(),
            canonical.replace(':', ""),
            format!("SHA256:{canonical}"),
            format!("  {} ", canonical.replace(':', " ")),
        ] {
            assert_eq!(parse_fingerprint(&typed).unwrap(), canonical, "{typed}");
        }
        for wrong in [
            "",
            "AB:CD",
            &canonical.replace("CD", "XY"),
            &(canonical.clone() + ":EF"),
        ] {
            assert!(parse_fingerprint(wrong).is_err(), "{wrong}");
        }
    }
}
