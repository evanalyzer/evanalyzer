# EVAnalyzer server

`evanalyzer server` lets EVAnalyzer clients on other machines log in and
work on this machine: projects, images and results are read and written
here, and the analysis runs here. Each logged-in user gets their own
`evanalyzer worker` process, confined to the folders that user may use.

```sh
evanalyzer server --config /etc/evanalyzer/server.toml
```

Clients connect with
`evanalyzer --remote wss://<host>:7400 --user <name> --remote-fingerprint <SHA256>`
(see [Encryption](#encryption-tls) for the fingerprint).

## Where settings come from

Each setting is taken from the first of these that sets it:

1. command-line arguments: `--listen`, `--session-store`, `--log-level`;
2. the `--config` file (TOML);
3. the built-in defaults.

The server reads no environment variables and no config file unless you
pass `--config`. It reads the files once at startup, so restart it after
editing them.

| File | What it is |
|---|---|
| [`server.toml`](server.toml) | Every setting with its default and an explanation. Copy it and keep the lines you change. |
| [`server-users.toml`](server-users.toml) | Example users file for `users.source = "file"`. |
| `crates/server/src/config.rs` | The reference: each setting is documented on its field. |

Unknown or misspelled keys and impossible values (a relative home folder, an
unsupported password format, …) stop the server at startup with an error
that names the key. The server's tests parse both example files, so the
examples always match the code.

## Users

`[users] source` chooses where accounts come from:

| `source` | Accounts | Home folder | Worker runs as |
|---|---|---|---|
| `single` (default) | One account in `[users.single]`: `admin` / `1234` until you change it (the server warns until you do) | `users.single.home`, default: the server account's home | the server's account |
| `linux` | This machine's system users, checked against `/etc/shadow`. The server needs root or membership in the `shadow` group. | from `/etc/passwd` | the user's own account, if the server runs as root |
| `file` | The `[[user]]` entries of a separate users file, `[users.file] path`. No system accounts needed. | `home` of each entry | the server's account |

### Allowed folders

A worker can read and write only its user's *allowed folders* and the
user's EVAnalyzer folder, which holds settings and templates. The allowed
folders come from:

- the account's own `allowed_dirs`: `users.single.allowed_dirs`, an
  `allowed_dirs` in the users file, or `[users.linux.overrides.<name>]`;
- otherwise `users.default_allowed_dirs`, which defaults to `["{home}"]`.

`{home}` stands for the user's home folder. For example, this gives
everyone their home plus a shared data folder:

```toml
[users]
default_allowed_dirs = ["{home}", "/data/microscopy"]
```

### Passwords

Create a password entry with EVAnalyzer itself:

```sh
evanalyzer hash-password            # asks twice, hidden; prints the hash
echo 'PASSWORD' | evanalyzer hash-password    # for scripts
```

It prints an Argon2id hash with a random salt and the OWASP-recommended
cost. Use it as the value of `password = "..."`. The config and users file
also accept hashes made with other tools:

| Format | Looks like | Generate with |
|---|---|---|
| Argon2id (recommended) | `$argon2id$v=19$m=…` | `evanalyzer hash-password` |
| bcrypt | `$2b$12$…` (also `$2a$`, `$2y$`) | `htpasswd -nbBC 12 "" 'PASSWORD' \| cut -d: -f2` (package `apache2-utils`) |
| yescrypt | `$y$…` | `mkpasswd -m yescrypt` (package `whois`) |
| sha512-crypt / sha256-crypt | `$6$…` / `$5$…` | `openssl passwd -6` |
| plain text | `plain:PASSWORD` | testing only: the server warns at startup |

Argon2i and Argon2d (`$argon2i$`, `$argon2d$`) are accepted too. Plain
text requires the `plain:` prefix, so a mistyped hash is rejected and can't
be mistaken for the password itself.

Anyone who can read the hashes can try to crack them offline, so make the
files readable only by the server's account:

```sh
chmod 600 /etc/evanalyzer/server.toml /etc/evanalyzer/users.toml
```

The server warns at startup if other accounts can read the users file.

## Encryption (TLS)

Connections between clients and the server are encrypted by default
(`wss://`). Workers listen on `127.0.0.1` only and are reached unencrypted.

**Without any setup** the server creates a self-signed certificate on first
start (in `tls.self_signed_dir`) and logs its fingerprint at every start:

```
Clients connect with wss:// - certificate fingerprint 1E:0F:D0:…:23:5D
```

Give that fingerprint to your users. They pass it once per connection with
`--remote-fingerprint`; the client then trusts exactly this certificate.
Without it, the client refuses to connect and prints the fingerprint it was
shown, so users can compare it with yours. If the fingerprint differs from
the expected one, the client refuses as well: either the certificate was
replaced (deleted `self_signed_dir`, new `cert`) or someone is intercepting
the connection. Keep `self_signed_dir` (include it in backups) and
the fingerprint stays the same.

For tests or a network you fully trust, clients can skip the check with
`--no-tls-verification`: still encrypted, but anyone in between could pose
as the server and read the password. The client logs a warning and its
status bar shows "server not verified".

**With a certificate from a public authority** (e.g. Let's Encrypt) set
`tls.cert` and `tls.key`. Clients connecting by that host name need no
fingerprint. A certificate from your organisation's own CA works too, with
the fingerprint, like a self-signed one.

**Without encryption**: `tls.enabled = false`, and clients use `ws://`. Only
for a server behind something that encrypts already (reverse proxy, VPN, SSH
tunnel) or for tests on one machine - the server warns when it listens on
the network without TLS, and the client's status bar shows "unencrypted".

## Workers

The server starts one worker per session and configures it with
arguments only:

```
evanalyzer worker --listen 127.0.0.1:<port> --token <token> \
    --home <user home> --root <allowed folder>... --log-level <log_level>
```

Its working directory is the user's home, and its environment is cleared
except for what the OS needs (`PATH`; on Windows also `SystemRoot`,
`windir`, `SystemDrive`, `TEMP`, `TMP`). Workers keep running when the
server restarts, and the server finds them again through
`session_store`.

## Example: a lab server with its own accounts

`/etc/evanalyzer/server.toml`:

```toml
listen = "0.0.0.0:7400"
log_level = "info"

[users]
source = "file"
default_allowed_dirs = ["{home}", "/data/microscopy"]

[users.file]
path = "/etc/evanalyzer/users.toml"
```

`/etc/evanalyzer/users.toml` (`chmod 600`):

```toml
[[user]]
id = "1"                       # stable - change the name, never the id
name = "alice"
password = "$argon2id$v=19$m=19456,t=2,p=1$…"
home = "/srv/evanalyzer/alice"

[[user]]
id = "2"
name = "bob"
password = "$2b$12$…"
home = "/srv/evanalyzer/bob"
allowed_dirs = ["{home}"]      # bob doesn't get /data/microscopy
```
