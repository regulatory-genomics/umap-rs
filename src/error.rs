//! Error types for the umap-rs crate.
//! Python exceptions (`ValueError` etc.) map onto this enum.

use thiserror::Error;

/// Errors raised by UMAP operations.
#[derive(Debug, Error)]
pub enum UmapError {
    /// Invalid input shape or invalid hyperparameter value.
    /// Python: `ValueError`.
    #[error("{0}")]
    InvalidInput(String),

    /// The input data is malformed (e.g. unsorted CSR, bad indices).
    #[error("{0}")]
    MalformedData(String),

    /// The algorithm could not proceed (e.g. spectral layout failure with no
    /// fallback available).
    #[error("{0}")]
    Computation(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, UmapError>;
