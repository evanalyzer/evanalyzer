//! Rust side of the in-app file browser (`dialogs/file_browser_dialog.slint`),
//! which replaces the OS open/save dialogs everywhere. It browses the
//! backend's file system ([`Backend::files`]), so in remote mode the user
//! picks files on the server - exactly the paths the server will open.
//!
//! Usage from a controller (on the UI thread):
//!
//! ```ignore
//! self.app_state.file_browser.open(
//!     FileRequest::save_file("Save project").filter("Project files", &["evaproj"]),
//!     move |path| {
//!         let Some(path) = path else { return }; // cancelled
//!         /* runs on the UI thread after the user confirmed */
//!     },
//! );
//! ```
//!
//! Folder listings run on a background thread, so a slow server never
//! freezes the UI.

use crate::{
    FileBrowserCrumb, FileBrowserEntry, FileBrowserMode, FileBrowserPlace, FileBrowserState,
};
use evanalyzer_app::backends::Backend;
use evanalyzer_app::fs::DirEntry;
use evanalyzer_app::fs::PlaceKind;
use evanalyzer_app::global::SUPPORTED_IMAGE_FORMATS;
use evanalyzer_cfg::{
    EVANALYZER_TRAINED_AI_MODELS, LEGACY_PROJECT_FILE_EXTENSION, PIPELINE_EXTENSIONS,
    PROJECT_FILE_EXTENSIONS, PROJECT_FILE_TEMPLATE_EXTENSIONS, RESULTS_FILE_EXTENSION,
};
use log::warn;
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileMode {
    OpenFile,
    OpenFolder,
    SaveFile,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileFilter {
    pub name: String,
    /// Lowercase, without the dot. Empty = every file.
    pub extensions: Vec<String>,
}

impl FileFilter {
    fn matches(&self, file_name: &str) -> bool {
        let lower = file_name.to_lowercase();
        self.extensions.is_empty()
            || self
                .extensions
                .iter()
                .any(|ext| lower.ends_with(&format!(".{ext}")))
    }
}

/// What to ask the user for. Built with the `open_file`/`open_folder`/
/// `save_file` constructors and the builder methods below.
#[derive(Clone, Debug)]
pub struct FileRequest {
    pub mode: FileMode,
    pub title: String,
    pub filters: Vec<FileFilter>,
    pub start_dir: Option<PathBuf>,
    pub file_name: Option<String>,
}

impl FileRequest {
    fn new(mode: FileMode, title: &str) -> Self {
        Self {
            mode,
            title: title.into(),
            filters: Vec::new(),
            start_dir: None,
            file_name: None,
        }
    }

    pub fn open_file(title: &str) -> Self {
        Self::new(FileMode::OpenFile, title)
    }

    pub fn open_folder(title: &str) -> Self {
        Self::new(FileMode::OpenFolder, title)
    }

    pub fn save_file(title: &str) -> Self {
        Self::new(FileMode::SaveFile, title)
    }

    /// Adds a file-type filter; the first one is active initially.
    pub fn filter(mut self, name: &str, extensions: &[&str]) -> Self {
        self.filters.push(FileFilter {
            name: name.into(),
            extensions: extensions.iter().map(|e| e.to_lowercase()).collect(),
        });
        self
    }

    /// Starts in `dir` (or its parent, for a file path) instead of the
    /// last-used folder.
    /// An empty path is ignored.
    pub fn start_in(mut self, dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        if !dir.as_os_str().is_empty() {
            self.start_dir = Some(dir);
        }
        self
    }

    /// Pre-filled file name (save mode).
    pub fn file_name(mut self, name: &str) -> Self {
        self.file_name = Some(name.into());
        self
    }

    fn accept_label(&self) -> &'static str {
        match self.mode {
            FileMode::OpenFile => "Open",
            FileMode::OpenFolder => "Select folder",
            FileMode::SaveFile => "Save",
        }
    }
}

type OnDone = Box<dyn FnOnce(Option<PathBuf>) + Send>;

#[derive(Default)]
struct Session {
    request: Option<FileRequest>,
    on_done: Option<OnDone>,
    current_dir: PathBuf,
    /// Everything in `current_dir`; `shown` is the filtered subset the rows
    /// display, in the same order as `FileBrowserState.entries`.
    listing: Vec<DirEntry>,
    shown: Vec<DirEntry>,
    selected: Option<usize>,
    filter: usize,
    search: String,
    show_hidden: bool,
    file_name: String,
    /// Bumped per navigation, so a slow listing that finishes after the
    /// user already moved on is ignored.
    generation: u64,
    /// Where the next dialog opens if the request doesn't say.
    last_dir: Option<PathBuf>,
    /// Save target waiting for the user to confirm replacing it.
    pending_overwrite: Option<PathBuf>,
}

/// One per window: the dialog is drawn inside that window.
pub struct FileBrowser<W: ComponentHandle + 'static> {
    window: slint::Weak<W>,
    backend: Arc<dyn Backend>,
    session: Mutex<Session>,
}

impl<W> FileBrowser<W>
where
    W: ComponentHandle + 'static,
    for<'a> FileBrowserState<'a>: slint::Global<'a, W>,
{
    pub fn new(window: slint::Weak<W>, backend: Arc<dyn Backend>) -> Self {
        Self {
            window,
            backend,
            session: Mutex::new(Session::default()),
        }
    }

    /// Wires the dialog's callbacks. Call once after the window exists.
    pub fn attach(self: &Arc<Self>, ui: &W) {
        let state = ui.global::<FileBrowserState>();
        let this = Arc::clone(self);
        state.on_navigate(move |path| this.navigate(PathBuf::from(path.as_str())));
        let this = Arc::clone(self);
        state.on_go_up(move || this.go_up());
        let this = Arc::clone(self);
        state.on_refresh(move || {
            let dir = this.session.lock().unwrap().current_dir.clone();
            this.navigate(dir);
        });
        let this = Arc::clone(self);
        state.on_select(move |index| this.select(index));
        let this = Arc::clone(self);
        state.on_activate(move |index| this.activate(index));
        let this = Arc::clone(self);
        state.on_accept(move || this.accept());
        let this = Arc::clone(self);
        state.on_cancel(move || this.close(None));
        let this = Arc::clone(self);
        state.on_search_edited(move |text| {
            this.session.lock().unwrap().search = text.to_string();
            this.refilter();
        });
        let this = Arc::clone(self);
        state.on_filter_selected(move |name| this.select_filter(name.as_str()));
        let this = Arc::clone(self);
        state.on_hidden_toggled(move |show| {
            this.session.lock().unwrap().show_hidden = show;
            this.with_state(|s| s.set_show_hidden(show));
            this.refilter();
        });
        let this = Arc::clone(self);
        state.on_path_entered(move |text| this.path_entered(text.trim()));
        let this = Arc::clone(self);
        state.on_create_folder(move |name| this.create_folder(name.trim()));
        let this = Arc::clone(self);
        state.on_overwrite_answered(move |replace| this.overwrite_answered(replace));
        let this = Arc::clone(self);
        state.on_file_name_edited(move |text| {
            this.session.lock().unwrap().file_name = text.to_string();
            this.update_can_accept();
        });
    }

    /// Shows the dialog. `on_done` runs once on the UI thread: with the
    /// chosen path, or `None` if the user cancelled (or there is no window,
    /// as in tests).
    pub fn open(
        self: &Arc<Self>,
        request: FileRequest,
        on_done: impl FnOnce(Option<PathBuf>) + Send + 'static,
    ) {
        if self.window.upgrade().is_none() {
            on_done(None);
            return;
        }
        // A second request replaces one still open: its caller hears "cancelled".
        if self.session.lock().unwrap().request.is_some() {
            self.close(None);
        }
        let start = {
            let mut s = self.session.lock().unwrap();
            s.filter = 0;
            s.search.clear();
            s.selected = None;
            s.pending_overwrite = None;
            s.file_name = request.file_name.clone().unwrap_or_default();
            s.listing.clear();
            s.shown.clear();
            let start = request.start_dir.clone().or_else(|| s.last_dir.clone());
            s.on_done = Some(Box::new(on_done));
            s.request = Some(request.clone());
            start
        };

        let location = if self.backend.is_remote() {
            format!("Files on server {}", self.backend.description())
        } else {
            "Files on this computer".to_string()
        };
        let remote = self.backend.is_remote();
        let filter_names: Vec<SharedString> = request
            .filters
            .iter()
            .map(|f| f.name.as_str().into())
            .collect();
        let first_filter = filter_names.first().cloned().unwrap_or_default();
        let file_name = request.file_name.clone().unwrap_or_default();
        self.with_state(|state| {
            state.set_mode(match request.mode {
                FileMode::OpenFile => FileBrowserMode::OpenFile,
                FileMode::OpenFolder => FileBrowserMode::OpenFolder,
                FileMode::SaveFile => FileBrowserMode::SaveFile,
            });
            state.set_title(request.title.as_str().into());
            state.set_accept_label(request.accept_label().into());
            state.set_location(location.into());
            state.set_remote(remote);
            state.set_filter_names(ModelRc::new(VecModel::from(filter_names)));
            state.set_filter_name(first_filter);
            state.set_search("".into());
            state.set_file_name(file_name.into());
            state.set_error("".into());
            state.set_overwrite_name("".into());
            state.set_entries(ModelRc::new(VecModel::from(Vec::<FileBrowserEntry>::new())));
            state.set_selected(-1);
            state.set_visible(true);
        });
        self.update_can_accept();

        // Places and the start folder load in the background.
        let this = Arc::clone(self);
        crate::helper::ui_thread::spawn(move || {
            let places = this.backend.files().places();
            let start_dir = match start {
                Some(dir) => this.resolve_start(dir),
                None => None,
            };
            let fallback = places
                .as_ref()
                .ok()
                .and_then(|p| p.first())
                .map(|p| p.path.clone());
            let ui_this = Arc::clone(&this);
            let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                match &places {
                    Ok(places) => {
                        let rows: Vec<FileBrowserPlace> = places
                            .iter()
                            .map(|p| FileBrowserPlace {
                                name: p.name.as_str().into(),
                                path: p.path.to_string_lossy().as_ref().into(),
                                kind: match p.kind {
                                    PlaceKind::Home => 0,
                                    PlaceKind::Drive => 1,
                                    PlaceKind::Folder => 2,
                                },
                            })
                            .collect();
                        ui_this.with_state(|s| s.set_places(ModelRc::new(VecModel::from(rows))));
                    }
                    Err(e) => ui_this.with_state(|s| s.set_error(e.to_string().into())),
                }
                if let Some(dir) = start_dir.or(fallback) {
                    ui_this.navigate(dir);
                }
            });
        });
    }

    /// A start folder that exists - the given path, its parent (for a file
    /// path), or nothing.
    fn resolve_start(&self, dir: PathBuf) -> Option<PathBuf> {
        let files = self.backend.files();
        match files.stat(&dir) {
            Ok(Some(entry)) if entry.is_dir => Some(dir),
            _ => {
                let parent = dir.parent()?.to_path_buf();
                matches!(files.stat(&parent), Ok(Some(e)) if e.is_dir).then_some(parent)
            }
        }
    }

    fn with_state(&self, f: impl FnOnce(FileBrowserState<'_>)) {
        if let Some(ui) = self.window.upgrade() {
            f(ui.global::<FileBrowserState>());
        }
    }

    fn navigate(self: &Arc<Self>, dir: PathBuf) {
        let generation = {
            let mut s = self.session.lock().unwrap();
            if s.request.is_none() {
                return;
            }
            s.generation += 1;
            s.current_dir = dir.clone();
            s.selected = None;
            s.generation
        };
        let crumbs = crumbs_for(&dir);
        self.with_state(|state| {
            state.set_current_path(dir.to_string_lossy().as_ref().into());
            state.set_crumbs(ModelRc::new(VecModel::from(crumbs)));
            state.set_loading(true);
            state.set_error("".into());
            state.set_selected(-1);
        });

        let this = Arc::clone(self);
        crate::helper::ui_thread::spawn(move || {
            let result = this.backend.files().list_dir(&dir);
            let ui_this = Arc::clone(&this);
            let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                {
                    let mut s = ui_this.session.lock().unwrap();
                    if s.generation != generation || s.request.is_none() {
                        return;
                    }
                    match &result {
                        Ok(entries) => s.listing = entries.clone(),
                        Err(_) => s.listing.clear(),
                    }
                }
                ui_this.with_state(|state| {
                    state.set_loading(false);
                    state.set_error(match &result {
                        Ok(_) => "".into(),
                        Err(e) => e.to_string().into(),
                    });
                });
                ui_this.refilter();
            });
        });
    }

    fn go_up(self: &Arc<Self>) {
        let parent = {
            let s = self.session.lock().unwrap();
            s.current_dir.parent().map(Path::to_path_buf)
        };
        if let Some(parent) = parent {
            self.navigate(parent);
        }
    }

    /// Recomputes the visible rows from the listing, filter and search.
    fn refilter(&self) {
        let (rows, status) = {
            let mut s = self.session.lock().unwrap();
            let Some(request) = &s.request else {
                return;
            };
            let mode = request.mode;
            let filter = request.filters.get(s.filter).cloned();
            let search = s.search.to_lowercase();
            let show_hidden = s.show_hidden;
            let shown: Vec<DirEntry> = s
                .listing
                .iter()
                .filter(|e| show_hidden || !e.name.starts_with('.'))
                .filter(|e| match mode {
                    FileMode::OpenFolder => e.is_dir,
                    _ => e.is_dir || filter.as_ref().is_none_or(|f| f.matches(&e.name)),
                })
                .filter(|e| search.is_empty() || e.name.to_lowercase().contains(&search))
                .cloned()
                .collect();
            s.shown = shown;
            s.selected = None;
            let rows: Vec<FileBrowserEntry> = s.shown.iter().map(entry_row).collect();
            (rows, summary(&s.shown))
        };
        self.with_state(|state| {
            state.set_entries(ModelRc::new(VecModel::from(rows)));
            state.set_selected(-1);
            state.set_status(status.into());
        });
        self.update_can_accept();
    }

    fn select_filter(&self, name: &str) {
        {
            let mut s = self.session.lock().unwrap();
            let Some(request) = &s.request else {
                return;
            };
            let Some(index) = request.filters.iter().position(|f| f.name == name) else {
                return;
            };
            s.filter = index;
        }
        self.with_state(|state| state.set_filter_name(name.into()));
        self.refilter();
    }

    fn select(&self, index: i32) {
        let (status, file_name) = {
            let mut s = self.session.lock().unwrap();
            let Some(entry) = usize::try_from(index)
                .ok()
                .and_then(|i| s.shown.get(i))
                .cloned()
            else {
                return;
            };
            s.selected = Some(index as usize);
            let saving = s
                .request
                .as_ref()
                .is_some_and(|r| r.mode == FileMode::SaveFile);
            let file_name = (saving && !entry.is_dir).then(|| entry.name.clone());
            if let Some(name) = &file_name {
                s.file_name = name.clone();
            }
            let status = if entry.is_dir {
                format!("{} - folder", entry.name)
            } else {
                format!("{} - {}", entry.name, format_size(entry.size))
            };
            (status, file_name)
        };
        self.with_state(|state| {
            state.set_selected(index);
            state.set_status(status.into());
            if let Some(name) = file_name {
                state.set_file_name(name.into());
            }
        });
        self.update_can_accept();
    }

    fn activate(self: &Arc<Self>, index: i32) {
        let entry = {
            let s = self.session.lock().unwrap();
            usize::try_from(index)
                .ok()
                .and_then(|i| s.shown.get(i))
                .cloned()
        };
        let Some(entry) = entry else {
            return;
        };
        if entry.is_dir {
            self.navigate(entry.path);
        } else {
            self.select(index);
            self.accept();
        }
    }

    fn update_can_accept(&self) {
        let can_accept = {
            let s = self.session.lock().unwrap();
            match s.request.as_ref().map(|r| r.mode) {
                Some(FileMode::OpenFile) => s.selected.is_some(),
                Some(FileMode::OpenFolder) => !s.current_dir.as_os_str().is_empty(),
                Some(FileMode::SaveFile) => valid_file_name(s.file_name.trim()),
                None => false,
            }
        };
        self.with_state(|state| state.set_can_accept(can_accept));
    }

    fn accept(self: &Arc<Self>) {
        let (mode, selected, current_dir, file_name, filter) = {
            let s = self.session.lock().unwrap();
            let Some(request) = &s.request else {
                return;
            };
            (
                request.mode,
                s.selected.and_then(|i| s.shown.get(i)).cloned(),
                s.current_dir.clone(),
                s.file_name.trim().to_string(),
                request.filters.get(s.filter).cloned(),
            )
        };
        match mode {
            FileMode::OpenFile => match selected {
                Some(entry) if entry.is_dir => self.navigate(entry.path),
                Some(entry) => self.close(Some(entry.path)),
                None => {}
            },
            FileMode::OpenFolder => {
                let dir = selected
                    .filter(|e| e.is_dir)
                    .map(|e| e.path)
                    .unwrap_or(current_dir);
                self.close(Some(dir));
            }
            FileMode::SaveFile => {
                if !valid_file_name(&file_name) {
                    return;
                }
                let name = with_default_extension(&file_name, filter.as_ref());
                let target = current_dir.join(&name);
                // Ask before replacing an existing file.
                let this = Arc::clone(self);
                crate::helper::ui_thread::spawn(move || {
                    let existing = this.backend.files().stat(&target);
                    let ui_this = Arc::clone(&this);
                    let _ =
                        crate::helper::ui_thread::invoke_from_event_loop(move || match existing {
                            Ok(Some(entry)) if entry.is_dir => ui_this.with_state(|s| {
                                s.set_error(format!("“{name}” is a folder").into())
                            }),
                            Ok(Some(_)) => {
                                ui_this.session.lock().unwrap().pending_overwrite = Some(target);
                                ui_this.with_state(|s| s.set_overwrite_name(name.into()));
                            }
                            Ok(None) => ui_this.close(Some(target)),
                            Err(e) => ui_this.with_state(|s| s.set_error(e.to_string().into())),
                        });
                });
            }
        }
    }

    fn overwrite_answered(self: &Arc<Self>, replace: bool) {
        let target = self.session.lock().unwrap().pending_overwrite.take();
        self.with_state(|s| s.set_overwrite_name("".into()));
        if let (true, Some(target)) = (replace, target) {
            self.close(Some(target));
        }
    }

    /// Typed into the path field: a folder opens, a file is selected (and
    /// accepted when opening files).
    fn path_entered(self: &Arc<Self>, text: &str) {
        if text.is_empty() {
            return;
        }
        let path = PathBuf::from(text);
        let this = Arc::clone(self);
        crate::helper::ui_thread::spawn(move || {
            let stat = this.backend.files().stat(&path);
            let ui_this = Arc::clone(&this);
            let _ = crate::helper::ui_thread::invoke_from_event_loop(move || match stat {
                Ok(Some(entry)) if entry.is_dir => ui_this.navigate(path),
                Ok(Some(_)) => {
                    let mode = ui_this
                        .session
                        .lock()
                        .unwrap()
                        .request
                        .as_ref()
                        .map(|r| r.mode);
                    if mode == Some(FileMode::OpenFile) {
                        ui_this.close(Some(path));
                    } else if let Some(parent) = path.parent() {
                        ui_this.navigate(parent.to_path_buf());
                    }
                }
                Ok(None) => ui_this.with_state(|s| {
                    s.set_error(format!("“{}” does not exist", path.display()).into())
                }),
                Err(e) => ui_this.with_state(|s| s.set_error(e.to_string().into())),
            });
        });
    }

    fn create_folder(self: &Arc<Self>, name: &str) {
        if !valid_file_name(name) {
            self.with_state(|s| s.set_error("Folder names can't contain / or \\".into()));
            return;
        }
        let target = self.session.lock().unwrap().current_dir.join(name);
        let this = Arc::clone(self);
        crate::helper::ui_thread::spawn(move || {
            let result = this.backend.files().create_dir_all(&target);
            let ui_this = Arc::clone(&this);
            let _ = crate::helper::ui_thread::invoke_from_event_loop(move || match result {
                Ok(()) => ui_this.navigate(target),
                Err(e) => ui_this.with_state(|s| s.set_error(e.to_string().into())),
            });
        });
    }

    /// Hides the dialog; runs the callback with `path` if one was chosen.
    fn close(self: &Arc<Self>, path: Option<PathBuf>) {
        let on_done = {
            let mut s = self.session.lock().unwrap();
            if !s.current_dir.as_os_str().is_empty() {
                s.last_dir = Some(s.current_dir.clone());
            }
            s.request = None;
            s.generation += 1;
            s.listing.clear();
            s.shown.clear();
            s.on_done.take()
        };
        self.with_state(|state| {
            state.set_visible(false);
            state.set_entries(ModelRc::new(VecModel::from(Vec::<FileBrowserEntry>::new())));
        });
        match on_done {
            Some(on_done) => on_done(path),
            None => {
                if let Some(path) = path {
                    warn!("File chosen with no one waiting for it: {}", path.display());
                }
            }
        }
    }
}

fn valid_file_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\'])
}

/// Appends the active filter's first extension if `name` has none of the
/// filter's extensions ("results" -> "results.evaproj").
fn with_default_extension(name: &str, filter: Option<&FileFilter>) -> String {
    match filter {
        Some(filter) if !filter.extensions.is_empty() && !filter.matches(name) => {
            format!("{name}.{}", filter.extensions[0])
        }
        _ => name.to_string(),
    }
}

fn crumbs_for(dir: &Path) -> Vec<FileBrowserCrumb> {
    let mut crumbs: Vec<FileBrowserCrumb> = dir
        .ancestors()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| FileBrowserCrumb {
            name: p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.to_string_lossy().into_owned())
                .as_str()
                .into(),
            path: p.to_string_lossy().as_ref().into(),
        })
        .collect();
    crumbs.reverse();
    crumbs
}

fn entry_kind(entry: &DirEntry) -> i32 {
    if entry.is_dir {
        return 0;
    }
    let lower = entry.name.to_lowercase();
    let has = |ext: &str| lower.ends_with(&format!(".{ext}"));
    if SUPPORTED_IMAGE_FORMATS.iter().any(|ext| has(ext)) {
        1
    } else if [
        PROJECT_FILE_EXTENSIONS,
        PROJECT_FILE_TEMPLATE_EXTENSIONS,
        PIPELINE_EXTENSIONS,
        LEGACY_PROJECT_FILE_EXTENSION,
    ]
    .iter()
    .any(|ext| has(ext))
    {
        2
    } else if has(EVANALYZER_TRAINED_AI_MODELS) {
        3
    } else if has(RESULTS_FILE_EXTENSION) {
        4
    } else {
        5
    }
}

fn entry_row(entry: &DirEntry) -> FileBrowserEntry {
    FileBrowserEntry {
        name: entry.name.as_str().into(),
        path: entry.path.to_string_lossy().as_ref().into(),
        is_dir: entry.is_dir,
        size: if entry.is_dir {
            "".into()
        } else {
            format_size(entry.size).into()
        },
        modified: entry.modified.map(format_time).unwrap_or_default().into(),
        kind: entry_kind(entry),
    }
}

fn summary(shown: &[DirEntry]) -> String {
    let folders = shown.iter().filter(|e| e.is_dir).count();
    let files = shown.len() - folders;
    let plural = |n: usize, word: &str| format!("{n} {word}{}", if n == 1 { "" } else { "s" });
    format!("{}, {}", plural(folders, "folder"), plural(files, "file"))
}

fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn format_time(seconds: i64) -> String {
    chrono::DateTime::from_timestamp(seconds, 0)
        .map(|t| {
            t.with_timezone(&chrono::Local)
                .format("%Y-%m-%d %H:%M")
                .to_string()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(name: &str) -> DirEntry {
        DirEntry {
            name: name.into(),
            path: PathBuf::from("/x").join(name),
            is_dir: false,
            size: 0,
            modified: None,
        }
    }

    #[test]
    fn filters_match_extensions_case_insensitively_including_double_ones() {
        let request = FileRequest::open_file("t").filter("Images", &["TIF", "ome.tif"]);
        let filter = &request.filters[0];
        assert!(filter.matches("a.tif"));
        assert!(filter.matches("A.TIF"));
        assert!(filter.matches("b.ome.tif"));
        assert!(!filter.matches("c.tiff"));
        assert!(FileRequest::open_file("t").filter("All", &[]).filters[0].matches("anything"));
    }

    #[test]
    fn save_names_get_the_filters_extension_only_when_missing() {
        let request = FileRequest::save_file("t").filter("Project", &["evaproj"]);
        let filter = request.filters.first();
        assert_eq!(with_default_extension("p", filter), "p.evaproj");
        assert_eq!(with_default_extension("p.evaproj", filter), "p.evaproj");
        assert_eq!(with_default_extension("p.EVAPROJ", filter), "p.EVAPROJ");
        assert_eq!(with_default_extension("p.txt", None), "p.txt");
    }

    #[test]
    fn file_names_with_separators_or_dot_names_are_invalid() {
        assert!(valid_file_name("results.evaproj"));
        for bad in ["", ".", "..", "a/b", "a\\b"] {
            assert!(!valid_file_name(bad), "{bad}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn crumbs_run_from_the_root_to_the_folder() {
        let crumbs = crumbs_for(Path::new("/data/plates/p1"));
        let names: Vec<_> = crumbs.iter().map(|c| c.name.to_string()).collect();
        let paths: Vec<_> = crumbs.iter().map(|c| c.path.to_string()).collect();
        assert_eq!(names, ["/", "data", "plates", "p1"]);
        assert_eq!(paths, ["/", "/data", "/data/plates", "/data/plates/p1"]);
    }

    #[test]
    fn sizes_and_kinds_are_human_readable() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(3 * 1024 * 1024 * 1024), "3.0 GB");
        assert_eq!(entry_kind(&file("a.czi")), 1);
        assert_eq!(entry_kind(&file("p.evaproj")), 2);
        assert_eq!(entry_kind(&file("m.evamodel")), 3);
        assert_eq!(entry_kind(&file("r.evadb")), 4);
        assert_eq!(entry_kind(&file("notes.txt")), 5);
    }

    #[test]
    fn the_summary_counts_folders_and_files() {
        let mut dir = file("d");
        dir.is_dir = true;
        assert_eq!(summary(&[dir, file("a"), file("b")]), "1 folder, 2 files");
    }

    // -- the dialog, driven through its UI callbacks -------------------------

    use crate::AppWindow;
    use crate::editor::test_support::test_ui_windows;
    use crate::helper::ui_thread::drain_ui_queue;
    use evanalyzer_app::backends::local::LocalBackend;
    use slint::Model;
    use std::sync::mpsc;

    /// A temp folder with a project, a text file, a hidden file and a
    /// subfolder, plus a browser attached to a real window.
    struct Fixture {
        dir: tempfile::TempDir,
        ui: AppWindow,
        browser: Arc<FileBrowser<AppWindow>>,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.evaproj"), "{}").unwrap();
        std::fs::write(dir.path().join("notes.txt"), "hello").unwrap();
        std::fs::write(dir.path().join(".hidden"), "").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub").join("b.evaproj"), "{}").unwrap();
        let (ui, _results_ui) = test_ui_windows();
        let browser = Arc::new(FileBrowser::new(
            ui.as_weak(),
            Arc::new(LocalBackend::default()) as Arc<dyn Backend>,
        ));
        browser.attach(&ui);
        Fixture { dir, ui, browser }
    }

    impl Fixture {
        fn state(&self) -> FileBrowserState<'_> {
            self.ui.global::<FileBrowserState>()
        }

        /// Opens `request` in the fixture folder; returns where the result goes.
        fn open(&self, request: FileRequest) -> mpsc::Receiver<Option<PathBuf>> {
            let (tx, rx) = mpsc::channel();
            self.browser
                .open(request.start_in(self.dir.path()), move |path| {
                    tx.send(path).unwrap()
                });
            drain_ui_queue();
            rx
        }

        fn shown(&self) -> Vec<String> {
            let entries = self.state().get_entries();
            (0..entries.row_count())
                .map(|i| entries.row_data(i).unwrap().name.to_string())
                .collect()
        }

        fn index_of(&self, name: &str) -> i32 {
            self.shown().iter().position(|n| n == name).unwrap() as i32
        }
    }

    fn sorted(mut names: Vec<String>) -> Vec<String> {
        names.sort();
        names
    }

    #[test]
    fn opening_shows_the_start_folder_filtered_and_without_hidden_files() {
        let f = fixture();
        let _rx = f.open(FileRequest::open_file("Open project").filter("Projects", &["evaproj"]));
        let state = f.state();
        assert!(state.get_visible());
        assert_eq!(state.get_title(), "Open project");
        assert_eq!(state.get_accept_label(), "Open");
        assert_eq!(state.get_location(), "Files on this computer");
        assert!(!state.get_remote());
        assert_eq!(
            state.get_current_path(),
            f.dir.path().to_string_lossy().as_ref()
        );
        assert!(state.get_crumbs().row_count() >= 1);
        assert!(state.get_places().row_count() >= 1);
        assert!(!state.get_loading());
        assert_eq!(sorted(f.shown()), ["a.evaproj", "sub"]);
        assert!(!state.get_can_accept(), "nothing selected yet");
        assert!(state.get_status().contains('1'), "{}", state.get_status());
    }

    #[test]
    fn hidden_files_search_and_filters_change_the_shown_rows() {
        let f = fixture();
        let _rx = f.open(
            FileRequest::open_file("Open")
                .filter("Projects", &["evaproj"])
                .filter("All files", &[]),
        );
        let state = f.state();
        assert_eq!(state.get_filter_names().row_count(), 2);

        state.invoke_filter_selected("All files".into());
        assert_eq!(state.get_filter_name(), "All files");
        assert_eq!(sorted(f.shown()), ["a.evaproj", "notes.txt", "sub"]);

        state.invoke_hidden_toggled(true);
        assert!(state.get_show_hidden());
        assert_eq!(f.shown().len(), 4);

        state.invoke_search_edited("NOTE".into());
        assert_eq!(f.shown(), ["notes.txt"]);

        // Unknown filter names are ignored.
        state.invoke_filter_selected("nope".into());
        assert_eq!(state.get_filter_name(), "All files");
    }

    #[test]
    fn opening_a_file_by_selecting_and_accepting_it() {
        let f = fixture();
        let rx = f.open(FileRequest::open_file("Open").filter("Projects", &["evaproj"]));
        let state = f.state();
        state.invoke_select(f.index_of("a.evaproj"));
        assert_eq!(state.get_selected(), f.index_of("a.evaproj"));
        assert!(state.get_can_accept());
        assert!(state.get_status().starts_with("a.evaproj - "));
        state.invoke_accept();
        assert_eq!(rx.try_recv().unwrap(), Some(f.dir.path().join("a.evaproj")));
        assert!(!state.get_visible());
    }

    #[test]
    fn activating_a_folder_enters_it_and_a_file_opens_it() {
        let f = fixture();
        let rx = f.open(FileRequest::open_file("Open").filter("Projects", &["evaproj"]));
        let state = f.state();
        state.invoke_select(f.index_of("sub"));
        assert_eq!(state.get_status(), "sub - folder");
        state.invoke_activate(f.index_of("sub"));
        drain_ui_queue();
        assert_eq!(f.shown(), ["b.evaproj"]);

        state.invoke_go_up();
        drain_ui_queue();
        assert!(f.shown().contains(&"a.evaproj".to_string()));
        state.invoke_navigate(f.dir.path().join("sub").to_string_lossy().as_ref().into());
        drain_ui_queue();
        state.invoke_refresh();
        drain_ui_queue();
        state.invoke_activate(0);
        assert_eq!(
            rx.try_recv().unwrap(),
            Some(f.dir.path().join("sub").join("b.evaproj"))
        );

        // Out-of-range rows do nothing.
        let _rx = f.open(FileRequest::open_file("Open"));
        state.invoke_select(99);
        state.invoke_activate(-1);
        assert_eq!(state.get_selected(), -1);
    }

    #[test]
    fn accepting_a_selected_folder_in_open_file_mode_enters_it() {
        let f = fixture();
        let rx = f.open(FileRequest::open_file("Open"));
        let state = f.state();
        state.invoke_accept(); // nothing selected: nothing happens
        state.invoke_select(f.index_of("sub"));
        state.invoke_accept();
        drain_ui_queue();
        assert!(rx.try_recv().is_err(), "still open");
        assert_eq!(f.shown(), ["b.evaproj"]);
    }

    #[test]
    fn picking_a_folder_returns_the_selected_or_current_one() {
        let f = fixture();
        let rx = f.open(FileRequest::open_folder("Choose folder"));
        let state = f.state();
        assert_eq!(state.get_accept_label(), "Select folder");
        assert_eq!(f.shown(), ["sub"], "only folders in folder mode");
        assert!(state.get_can_accept());
        state.invoke_select(0);
        state.invoke_accept();
        assert_eq!(rx.try_recv().unwrap(), Some(f.dir.path().join("sub")));

        let rx = f.open(FileRequest::open_folder("Choose folder"));
        f.state().invoke_accept();
        assert_eq!(rx.try_recv().unwrap(), Some(f.dir.path().to_path_buf()));
    }

    #[test]
    fn saving_appends_the_extension_and_asks_before_replacing() {
        let f = fixture();
        let rx = f.open(
            FileRequest::save_file("Save project")
                .filter("Projects", &["evaproj"])
                .file_name("new"),
        );
        let state = f.state();
        assert_eq!(state.get_accept_label(), "Save");
        assert_eq!(state.get_file_name(), "new");
        assert!(state.get_can_accept());
        state.invoke_accept();
        drain_ui_queue();
        assert_eq!(
            rx.try_recv().unwrap(),
            Some(f.dir.path().join("new.evaproj"))
        );

        // Existing file: asks first. "No" keeps the dialog open.
        let rx = f.open(FileRequest::save_file("Save").filter("Projects", &["evaproj"]));
        state.invoke_select(f.index_of("a.evaproj"));
        assert_eq!(
            state.get_file_name(),
            "a.evaproj",
            "selecting a file fills the name"
        );
        state.invoke_accept();
        drain_ui_queue();
        assert_eq!(state.get_overwrite_name(), "a.evaproj");
        state.invoke_overwrite_answered(false);
        assert_eq!(state.get_overwrite_name(), "");
        assert!(rx.try_recv().is_err());
        state.invoke_accept();
        drain_ui_queue();
        state.invoke_overwrite_answered(true);
        assert_eq!(rx.try_recv().unwrap(), Some(f.dir.path().join("a.evaproj")));
    }

    #[test]
    fn saving_over_a_folder_or_with_an_invalid_name_is_refused() {
        let f = fixture();
        let rx = f.open(FileRequest::save_file("Save"));
        let state = f.state();
        state.invoke_file_name_edited("sub".into());
        state.invoke_accept();
        drain_ui_queue();
        assert!(state.get_error().contains("is a folder"));

        state.invoke_file_name_edited("a/b".into());
        assert!(!state.get_can_accept());
        state.invoke_accept();
        drain_ui_queue();
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn typing_a_path_opens_folders_and_files_and_reports_missing_ones() {
        let f = fixture();
        let rx = f.open(FileRequest::open_file("Open"));
        let state = f.state();
        let typed = |p: PathBuf| state.invoke_path_entered(p.to_string_lossy().as_ref().into());

        state.invoke_path_entered("".into());
        typed(f.dir.path().join("missing.txt"));
        drain_ui_queue();
        assert!(state.get_error().contains("does not exist"));

        typed(f.dir.path().join("sub"));
        drain_ui_queue();
        assert_eq!(f.shown(), ["b.evaproj"]);

        typed(f.dir.path().join("notes.txt"));
        drain_ui_queue();
        assert_eq!(rx.try_recv().unwrap(), Some(f.dir.path().join("notes.txt")));

        // In save mode a typed file only navigates to its folder.
        let rx = f.open(FileRequest::save_file("Save"));
        typed(f.dir.path().join("sub").join("b.evaproj"));
        drain_ui_queue();
        assert_eq!(f.shown(), ["b.evaproj"]);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn creating_a_folder_enters_it_and_bad_names_are_refused() {
        let f = fixture();
        let _rx = f.open(FileRequest::open_folder("Choose"));
        let state = f.state();
        state.invoke_create_folder("new folder".into());
        drain_ui_queue();
        assert!(f.dir.path().join("new folder").is_dir());
        assert!(state.get_current_path().ends_with("new folder"));

        state.invoke_create_folder("a/b".into());
        assert!(state.get_error().contains("can't contain"));
    }

    #[test]
    fn cancelling_or_a_second_request_reports_no_path() {
        let f = fixture();
        let first = f.open(FileRequest::open_file("First"));
        let second = f.open(FileRequest::open_file("Second"));
        assert_eq!(first.try_recv().unwrap(), None, "replaced by the second");
        f.state().invoke_cancel();
        assert_eq!(second.try_recv().unwrap(), None);
        assert!(!f.state().get_visible());

        // Navigating with no dialog open is ignored.
        f.state()
            .invoke_navigate(f.dir.path().to_string_lossy().as_ref().into());
        drain_ui_queue();
        assert!(f.shown().is_empty());
    }

    #[test]
    fn the_next_dialog_starts_where_the_last_one_ended() {
        let f = fixture();
        let rx = f.open(FileRequest::open_file("Open"));
        f.state()
            .invoke_navigate(f.dir.path().join("sub").to_string_lossy().as_ref().into());
        drain_ui_queue();
        f.state().invoke_cancel();
        assert_eq!(rx.try_recv().unwrap(), None);

        let (tx, _rx) = mpsc::channel();
        f.browser.open(FileRequest::open_file("Again"), move |p| {
            tx.send(p).unwrap()
        });
        drain_ui_queue();
        assert_eq!(f.shown(), ["b.evaproj"]);
    }

    #[test]
    fn a_start_path_pointing_at_a_file_opens_its_folder() {
        let f = fixture();
        let (tx, _rx) = mpsc::channel();
        f.browser.open(
            FileRequest::open_file("Open").start_in(f.dir.path().join("sub").join("b.evaproj")),
            move |p| tx.send(p).unwrap(),
        );
        drain_ui_queue();
        assert_eq!(f.shown(), ["b.evaproj"]);
    }

    #[test]
    fn listing_a_missing_folder_shows_the_error() {
        let f = fixture();
        let _rx = f.open(FileRequest::open_file("Open"));
        f.state()
            .invoke_navigate(f.dir.path().join("gone").to_string_lossy().as_ref().into());
        drain_ui_queue();
        assert!(!f.state().get_error().is_empty());
        assert!(f.shown().is_empty());
    }
}
