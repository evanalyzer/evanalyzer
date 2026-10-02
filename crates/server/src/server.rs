use crate::{session_management::SessionManagement, user_management::UserManagement};
use log::info;
use std::sync::Arc;

struct Server {
    user_management: Arc<dyn UserManagement>,
    session_management: SessionManagement,
}

pub fn serve(listen: String) {
    info!("Starting server");
}

pub fn listen_for_login() {}
