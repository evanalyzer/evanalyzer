use std::time::SystemTime;

pub struct SessionManagement {}

pub struct Session {
    pub start_date: SystemTime,
    pub session_token: String,
}

impl SessionManagement {
    pub fn new() -> Self {
        return Self {};
    }

    pub fn open_or_create_session(username: String) -> Session {
        return Session {
            start_date: todo!(),
            session_token: todo!(),
        };
    }

    fn create_session() -> Session {
        return Session {
            start_date: todo!(),
            session_token: todo!(),
        };
    }
    fn restore_session() -> Session {
        return Session {
            start_date: todo!(),
            session_token: todo!(),
        };
    }
    fn close_session(session_token: String) {}
}
