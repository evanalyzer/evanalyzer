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
                unix_account: None,
            });
        }

        return super::AuthenticationStatus::PasswordWrong;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::user_management::AuthenticationStatus;

    #[test]
    fn only_the_configured_name_and_password_log_in() {
        let users = SingleUser::default();
        match users.login("admin".into(), "1234".into()) {
            AuthenticationStatus::Authenticated(user) => {
                assert_eq!(user.username, "admin");
                assert_eq!(user.user_id, "user-id");
                assert!(user.unix_account.is_none());
            }
            _ => panic!("expected a login"),
        }
        for (name, password) in [("admin", "12345"), ("root", "1234"), ("", "")] {
            assert!(matches!(
                users.login(name.into(), password.into()),
                AuthenticationStatus::PasswordWrong
            ));
        }
    }
}
