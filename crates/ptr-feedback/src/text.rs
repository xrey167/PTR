//! The text view of a candidate value, shared by clustering, persistence and prompting.

use std::borrow::Cow;

/// A value that has a canonical text form. Clustering, trajectory records and prompt templates
/// all read candidates through it, so the caller states once how its `T` is rendered.
pub trait CandidateText {
    fn candidate_text(&self) -> Cow<'_, str>;
}

impl CandidateText for str {
    fn candidate_text(&self) -> Cow<'_, str> {
        Cow::Borrowed(self)
    }
}

impl CandidateText for String {
    fn candidate_text(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.as_str())
    }
}

impl<T: CandidateText + ?Sized> CandidateText for &T {
    fn candidate_text(&self) -> Cow<'_, str> {
        (**self).candidate_text()
    }
}

/// A value that can be built from the text a model returned. The conversion is total: output
/// that does not parse must still yield a value (which the verifier then rejects), because a
/// [`crate::Generator`] cannot fail.
pub trait ModelOutput: CandidateText + Sized {
    fn from_model_text(text: &str) -> Self;
}

impl ModelOutput for String {
    fn from_model_text(text: &str) -> Self {
        text.to_owned()
    }
}

/// Cut `text` to at most `max_chars` characters on a character boundary.
pub(crate) fn truncate_chars(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((end, _)) => &text[..end],
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_and_references_render_as_themselves() {
        assert_eq!("abc".candidate_text(), "abc");
        assert_eq!(String::from("x").candidate_text(), "x");
        assert_eq!((&"y").candidate_text(), "y");
    }

    #[test]
    fn truncation_respects_character_boundaries() {
        assert_eq!(truncate_chars("héllo", 2), "hé");
        assert_eq!(truncate_chars("abc", 10), "abc");
        assert_eq!(truncate_chars("abc", 0), "");
    }
}
