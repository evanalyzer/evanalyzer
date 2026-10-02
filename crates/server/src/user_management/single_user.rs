use crate::user_management::UserManagement;

pub struct SingleUser {
    pub username: String,
    pub password: String,
    pub userid: String,
}

impl SingleUser {
    pub fn default() -> Self {
        Self {
            username: "admin".into(),
            password: "1234".into(),
            userid: "user-id".into(),
        }
    }
}

impl UserManagement for SingleUser {
    fn login(&self, username: String, password: String) -> super::AuthenticationStatus {
        if username == self.username && password == self.password {
            return super::AuthenticationStatus::Authenticated(super::User {
                user_id: self.userid.clone(),
                username,
            });
        }

        return super::AuthenticationStatus::PasswordWrong;
    }
}
