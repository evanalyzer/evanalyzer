pub mod linux_users;
pub mod single_user;

pub struct User {
    pub user_id: String,
    pub username: String,
}

pub enum AuthenticationStatus {
    Authenticated(User),
    UserNotFound,
    PasswordWrong,
}

pub trait UserManagement: Send + Sync {
    fn login(&self, username: String, password: String) -> AuthenticationStatus;
}
