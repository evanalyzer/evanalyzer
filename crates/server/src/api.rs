use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Request {
    Login {
        username: String,
        password: String,
    },
    /// Back to the worker of an earlier login, after the connection dropped:
    /// the session token from that login instead of the password. Valid as
    /// long as that worker runs.
    Resume {
        session_token: String,
    },
    Exit,
}

#[derive(Serialize, Deserialize)]
pub enum ResponseState {
    Accepted,
    Error,
}

#[derive(Serialize, Deserialize)]
pub struct Response {
    pub response: ResponseState,
    pub msg: String,
    /// Handed out at login; identifies the user's session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_token: Option<String>,
}

impl Response {
    pub fn accepted(msg: impl Into<String>) -> Self {
        Self {
            response: ResponseState::Accepted,
            msg: msg.into(),
            session_token: None,
        }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            response: ResponseState::Error,
            msg: msg.into(),
            session_token: None,
        }
    }
}
