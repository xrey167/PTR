//! Text-model adaptors for the repair loop.
//!
//! [`PromptedGenerator`] and [`PromptedRepairer`] implement [`Generator`] and [`Repairer`] over
//! any [`TextModel`], a one-method text completion trait defined here so this crate depends on
//! no model crate. They fill a fixed prompt template, call the model with a token limit, and cut
//! its answer to a character bound. They make the loop drivable by a model; they say nothing
//! about whether a model is any good, and no trained model ships with them. The verifier, not
//! the model, decides what is accepted.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

use crate::candidate::{Candidate, Critique};
use crate::repair::{Generator, Repairer};
use crate::text::{truncate_chars, ModelOutput};

/// Most characters of a critique issue code placed in a prompt.
const MAX_ISSUE_CHARS: usize = 128;
/// Most issues placed in a prompt.
const MAX_PROMPT_ISSUES: usize = 32;
/// Most prompts a [`ScriptedTextModel`] remembers.
const MAX_RECORDED_PROMPTS: usize = 1024;

/// Why a text model produced no answer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextModelError {
    /// The model cannot be reached or is not loaded.
    Unavailable,
    /// The model declined to answer.
    Refused,
    /// A scripted model has no reply left.
    ScriptExhausted,
}

impl fmt::Display for TextModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Unavailable => "the text model is unavailable",
            Self::Refused => "the text model refused",
            Self::ScriptExhausted => "the scripted text model has no reply left",
        })
    }
}

impl Error for TextModelError {}

/// Completes a prompt with at most `max_tokens` tokens of text.
pub trait TextModel {
    /// # Errors
    /// Returns [`TextModelError`] when no answer is produced.
    fn complete(&self, prompt: &str, max_tokens: u32) -> Result<String, TextModelError>;
}

impl<M: TextModel + ?Sized> TextModel for &M {
    fn complete(&self, prompt: &str, max_tokens: u32) -> Result<String, TextModelError> {
        (**self).complete(prompt, max_tokens)
    }
}

/// A deterministic model for tests: it returns its scripted replies in order and records every
/// prompt it was given.
#[derive(Debug, Default)]
pub struct ScriptedTextModel {
    replies: RefCell<VecDeque<Result<String, TextModelError>>>,
    prompts: RefCell<Vec<(String, u32)>>,
}

impl ScriptedTextModel {
    pub fn new(replies: impl IntoIterator<Item = Result<String, TextModelError>>) -> Self {
        Self {
            replies: RefCell::new(replies.into_iter().collect()),
            prompts: RefCell::new(Vec::new()),
        }
    }

    /// A script of successful replies.
    pub fn from_texts<S: Into<String>>(texts: impl IntoIterator<Item = S>) -> Self {
        Self::new(texts.into_iter().map(|text| Ok(text.into())))
    }

    /// The prompts received so far with their token limits (at most 1024 are kept).
    pub fn prompts(&self) -> Vec<(String, u32)> {
        self.prompts.borrow().clone()
    }
}

impl TextModel for ScriptedTextModel {
    fn complete(&self, prompt: &str, max_tokens: u32) -> Result<String, TextModelError> {
        let mut prompts = self.prompts.borrow_mut();
        if prompts.len() < MAX_RECORDED_PROMPTS {
            prompts.push((prompt.to_owned(), max_tokens));
        }
        self.replies
            .borrow_mut()
            .pop_front()
            .unwrap_or(Err(TextModelError::ScriptExhausted))
    }
}

/// Why an adaptor could not be built.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromptError {
    NoTokens,
    NoOutputChars,
    /// The prompt budget cannot hold even the template.
    PromptBudgetTooSmall,
}

impl fmt::Display for PromptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NoTokens => "max_tokens must be positive",
            Self::NoOutputChars => "max_output_chars must be positive",
            Self::PromptBudgetTooSmall => "max_prompt_chars is too small for the template",
        })
    }
}

impl Error for PromptError {}

/// Bounds on one model call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromptOptions {
    /// Token limit passed to the model.
    pub max_tokens: u32,
    /// The model's answer is cut to this many characters.
    pub max_output_chars: usize,
    /// Whole-prompt cap in characters: the task, previous candidate and issue list are cut so the
    /// finished prompt, template included, never exceeds it.
    pub max_prompt_chars: usize,
}

impl PromptOptions {
    fn validate(&self) -> Result<(), PromptError> {
        if self.max_tokens == 0 {
            return Err(PromptError::NoTokens);
        }
        if self.max_output_chars == 0 {
            return Err(PromptError::NoOutputChars);
        }
        if self.max_prompt_chars < 256 {
            return Err(PromptError::PromptBudgetTooSmall);
        }
        Ok(())
    }
}

/// What an adaptor has done so far.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PromptedStats {
    pub calls: u32,
    /// Calls where the model returned an error.
    pub model_errors: u32,
    /// Calls that returned only whitespace.
    pub empty_outputs: u32,
    /// The last model error, if any.
    pub last_error: Option<TextModelError>,
}

/// Line that stands for issues left out of a repair prompt.
const ISSUES_OMITTED_LINE: &str = "- (further issues omitted)\n";

fn generation_text(task: &str, draw: u32, task_budget: usize) -> String {
    format!(
        "You propose one candidate solution.\n\nTask:\n{}\n\nDraw number: {draw}\n\
         Reply with the candidate only, no commentary.",
        truncate_chars(task, task_budget)
    )
}

fn repair_text(task: &str, previous: &str, listed: &str, budget: usize) -> String {
    format!(
        "You repair one candidate solution.\n\nTask:\n{}\n\nPrevious candidate:\n{}\n\n\
         The verifier reported these issues:\n{listed}\n\
         Reply with the corrected candidate only, no commentary.",
        truncate_chars(task, budget),
        truncate_chars(previous, budget)
    )
}

/// The issue list, at most `budget` characters: lines that do not fit are replaced by a marker.
fn list_issues(issues: &[String], budget: usize) -> String {
    let marker = ISSUES_OMITTED_LINE.chars().count();
    let mut listed = String::new();
    let mut used = 0;
    for (index, issue) in issues.iter().take(MAX_PROMPT_ISSUES).enumerate() {
        let line = format!("- {}\n", truncate_chars(issue, MAX_ISSUE_CHARS));
        let more = index + 1 < issues.len();
        let width = line.chars().count();
        if used + width + if more { marker } else { 0 } > budget {
            break;
        }
        used += width;
        listed.push_str(&line);
    }
    let dropped = issues.len() > MAX_PROMPT_ISSUES || listed.lines().count() < issues.len();
    if dropped && used + marker <= budget {
        listed.push_str(ISSUES_OMITTED_LINE);
    }
    listed
}

/// The generation prompt for the `draw`-th (0-based) request, at most
/// `options.max_prompt_chars` characters.
pub fn generation_prompt(task: &str, draw: u32, options: &PromptOptions) -> String {
    let max = options.max_prompt_chars;
    let prompt = generation_text(task, draw, max);
    let over = prompt.chars().count().saturating_sub(max);
    if over == 0 {
        return prompt;
    }
    let task_chars = task.chars().count().min(max);
    let shrunk = generation_text(task, draw, task_chars.saturating_sub(over));
    truncate_chars(&shrunk, max).to_owned()
}

/// The repair prompt for `previous` and the verifier issues named by `issues`, at most
/// `options.max_prompt_chars` characters. A prompt that fits the budget with the task and the
/// previous candidate each cut to half of it is returned as built; otherwise the template's
/// fixed text is subtracted from the budget and the rest is split between the issue list (a
/// third, with a marker for dropped issues), the task and the previous candidate.
pub fn repair_prompt(
    task: &str,
    previous: &str,
    issues: &[String],
    options: &PromptOptions,
) -> String {
    let max = options.max_prompt_chars;
    let first = repair_text(task, previous, &list_issues(issues, usize::MAX), max / 2);
    if first.chars().count() <= max {
        return first;
    }
    let free = max.saturating_sub(repair_text("", "", "", 0).chars().count());
    let listed = list_issues(issues, free / 3);
    let rest = free.saturating_sub(listed.chars().count());
    let shrunk = repair_text(task, previous, &listed, rest / 2);
    truncate_chars(&shrunk, max).to_owned()
}

fn clean(answer: &str, options: &PromptOptions) -> String {
    truncate_chars(answer.trim(), options.max_output_chars).to_owned()
}

fn call(
    model: &dyn TextModel,
    prompt: &str,
    options: &PromptOptions,
    stats: &mut PromptedStats,
) -> Option<String> {
    stats.calls += 1;
    match model.complete(prompt, options.max_tokens) {
        Ok(answer) => {
            let cleaned = clean(&answer, options);
            if cleaned.is_empty() {
                stats.empty_outputs += 1;
                None
            } else {
                Some(cleaned)
            }
        }
        Err(error) => {
            stats.model_errors += 1;
            stats.last_error = Some(error);
            None
        }
    }
}

/// A [`Generator`] that asks a [`TextModel`] for a candidate. When the model errors or answers
/// with nothing, it yields `T::from_model_text("")`, which the verifier judges like any other
/// value; the failure is counted in [`PromptedGenerator::stats`].
pub struct PromptedGenerator<M, T> {
    model: M,
    task: String,
    options: PromptOptions,
    draws: u32,
    stats: PromptedStats,
    output: PhantomData<fn() -> T>,
}

impl<M: TextModel, T: ModelOutput> PromptedGenerator<M, T> {
    /// # Errors
    /// Returns [`PromptError`] for invalid options.
    pub fn new(model: M, task: &str, options: PromptOptions) -> Result<Self, PromptError> {
        options.validate()?;
        Ok(Self {
            model,
            task: task.to_owned(),
            options,
            draws: 0,
            stats: PromptedStats::default(),
            output: PhantomData,
        })
    }

    pub fn stats(&self) -> PromptedStats {
        self.stats
    }

    pub fn model(&self) -> &M {
        &self.model
    }
}

impl<M: TextModel, T: ModelOutput> Generator<T> for PromptedGenerator<M, T> {
    fn generate(&mut self) -> T {
        let prompt = generation_prompt(&self.task, self.draws, &self.options);
        self.draws += 1;
        let answer = call(&self.model, &prompt, &self.options, &mut self.stats);
        T::from_model_text(answer.as_deref().unwrap_or(""))
    }
}

/// A [`Repairer`] that shows a [`TextModel`] the rejected candidate and the verifier's issues.
/// It declines (returns `None`, so the loop draws afresh) when the model errors or answers with
/// nothing.
pub struct PromptedRepairer<M, T> {
    model: M,
    task: String,
    options: PromptOptions,
    stats: PromptedStats,
    output: PhantomData<fn() -> T>,
}

impl<M: TextModel, T: ModelOutput> PromptedRepairer<M, T> {
    /// # Errors
    /// Returns [`PromptError`] for invalid options.
    pub fn new(model: M, task: &str, options: PromptOptions) -> Result<Self, PromptError> {
        options.validate()?;
        Ok(Self {
            model,
            task: task.to_owned(),
            options,
            stats: PromptedStats::default(),
            output: PhantomData,
        })
    }

    pub fn stats(&self) -> PromptedStats {
        self.stats
    }

    pub fn model(&self) -> &M {
        &self.model
    }
}

impl<M: TextModel, T: ModelOutput> Repairer<T> for PromptedRepairer<M, T> {
    fn repair(&mut self, previous: &Candidate<T>, critique: &Critique) -> Option<T> {
        let text = previous.value.candidate_text();
        let prompt = repair_prompt(&self.task, &text, &critique.issues, &self.options);
        call(&self.model, &prompt, &self.options, &mut self.stats)
            .map(|answer| T::from_model_text(&answer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::CandidateId;

    fn options() -> PromptOptions {
        PromptOptions {
            max_tokens: 64,
            max_output_chars: 16,
            max_prompt_chars: 1000,
        }
    }

    #[test]
    fn invalid_options_are_typed_errors() {
        let model = ScriptedTextModel::default();
        let build = |o: PromptOptions| PromptedGenerator::<_, String>::new(&model, "t", o).err();
        assert_eq!(
            build(PromptOptions {
                max_tokens: 0,
                ..options()
            }),
            Some(PromptError::NoTokens)
        );
        assert_eq!(
            build(PromptOptions {
                max_output_chars: 0,
                ..options()
            }),
            Some(PromptError::NoOutputChars)
        );
        assert_eq!(
            build(PromptOptions {
                max_prompt_chars: 10,
                ..options()
            }),
            Some(PromptError::PromptBudgetTooSmall)
        );
        let repairer = PromptedRepairer::<_, String>::new(
            &model,
            "t",
            PromptOptions {
                max_tokens: 0,
                ..options()
            },
        );
        assert_eq!(repairer.err(), Some(PromptError::NoTokens));
    }

    #[test]
    fn the_scripted_model_replays_records_and_runs_out() {
        let model = ScriptedTextModel::new([Ok("one".into()), Err(TextModelError::Refused)]);
        assert_eq!(model.complete("p1", 5).unwrap(), "one");
        assert_eq!(model.complete("p2", 6), Err(TextModelError::Refused));
        assert_eq!(
            model.complete("p3", 7),
            Err(TextModelError::ScriptExhausted)
        );
        assert_eq!(
            model.prompts(),
            [
                ("p1".to_owned(), 5),
                ("p2".to_owned(), 6),
                ("p3".to_owned(), 7)
            ]
        );
    }

    #[test]
    fn the_generator_bounds_and_trims_output_and_counts_draws() {
        let model = ScriptedTextModel::from_texts(["  hello world, this is long  ", "second"]);
        let mut generator =
            PromptedGenerator::<_, String>::new(&model, "write", options()).unwrap();
        assert_eq!(generator.generate(), "hello world, thi");
        assert_eq!(generator.generate(), "second");
        let prompts = model.prompts();
        assert!(prompts[0].0.contains("Draw number: 0") && prompts[1].0.contains("Draw number: 1"));
        assert!(prompts[0].0.contains("write"));
        assert_eq!(prompts[0].1, 64);
        assert_eq!(generator.stats().calls, 2);
    }

    #[test]
    fn model_failures_yield_an_empty_value_and_are_counted() {
        let model = ScriptedTextModel::new([Err(TextModelError::Unavailable), Ok("   ".into())]);
        let mut generator = PromptedGenerator::<_, String>::new(&model, "t", options()).unwrap();
        assert_eq!(generator.generate(), "");
        assert_eq!(generator.generate(), "");
        let stats = generator.stats();
        assert_eq!(
            (stats.calls, stats.model_errors, stats.empty_outputs),
            (2, 1, 1)
        );
        assert_eq!(stats.last_error, Some(TextModelError::Unavailable));
    }

    #[test]
    fn the_repair_prompt_carries_the_previous_candidate_and_the_issues() {
        let model = ScriptedTextModel::from_texts(["fixed"]);
        let mut repairer = PromptedRepairer::<_, String>::new(&model, "task!", options()).unwrap();
        let previous = Candidate {
            id: CandidateId::from("c0"),
            value: "broken text".to_owned(),
            verification: None,
        };
        let critique = Critique {
            candidate: previous.id.clone(),
            issues: vec!["target".into(), "effect".into()],
            repairable: true,
        };
        assert_eq!(
            repairer.repair(&previous, &critique).as_deref(),
            Some("fixed")
        );
        let prompt = &model.prompts()[0].0;
        for needle in ["task!", "broken text", "- target", "- effect"] {
            assert!(prompt.contains(needle), "{needle}");
        }
    }

    #[test]
    fn the_repairer_declines_when_the_model_fails_or_says_nothing() {
        let model = ScriptedTextModel::new([Err(TextModelError::Refused), Ok("".into())]);
        let mut repairer = PromptedRepairer::<_, String>::new(&model, "t", options()).unwrap();
        let previous = Candidate {
            id: CandidateId::from("c0"),
            value: "x".to_owned(),
            verification: None,
        };
        let critique = Critique {
            candidate: previous.id.clone(),
            issues: vec![],
            repairable: true,
        };
        assert_eq!(repairer.repair(&previous, &critique), None);
        assert_eq!(repairer.repair(&previous, &critique), None);
        assert_eq!(
            repairer.repair(&previous, &critique),
            None,
            "script exhausted"
        );
        assert_eq!(repairer.stats().model_errors, 2);
    }

    #[test]
    fn prompts_are_bounded_whatever_the_inputs() {
        let o = PromptOptions {
            max_prompt_chars: 300,
            ..options()
        };
        let issues: Vec<String> = (0..100)
            .map(|i| format!("{i}{}", "x".repeat(500)))
            .collect();
        let prompt = repair_prompt(&"t".repeat(10_000), &"p".repeat(10_000), &issues, &o);
        assert!(prompt.chars().count() < 300 + 32 * (MAX_ISSUE_CHARS + 3) + 400);
        assert!(prompt.contains("further issues omitted"));
        assert!(
            generation_prompt(&"t".repeat(10_000), 0, &o)
                .chars()
                .count()
                < 500
        );
    }

    #[test]
    fn the_repairer_never_sends_more_than_the_prompt_budget() {
        let o = PromptOptions {
            max_prompt_chars: 256,
            ..options()
        };
        let model = ScriptedTextModel::from_texts(["fixed"]);
        let mut repairer =
            PromptedRepairer::<_, String>::new(&model, &"t".repeat(5000), o).unwrap();
        let previous = Candidate {
            id: CandidateId::from("c0"),
            value: "p".repeat(5000),
            verification: None,
        };
        let critique = Critique {
            candidate: previous.id.clone(),
            issues: (0..32).map(|i| format!("{i}{}", "x".repeat(500))).collect(),
            repairable: true,
        };
        assert!(repairer.repair(&previous, &critique).is_some());
        let prompt = &model.prompts()[0].0;
        assert!(
            prompt.chars().count() <= 256,
            "{} chars",
            prompt.chars().count()
        );
    }

    #[test]
    fn prompts_fit_the_budget_for_every_small_budget_and_multibyte_text() {
        let issues: Vec<String> = (0..40)
            .map(|i| format!("é{i}{}", "ß".repeat(300)))
            .collect();
        for max in [256usize, 300, 400, 777, 1000, 4000] {
            let o = PromptOptions {
                max_prompt_chars: max,
                ..options()
            };
            let r = repair_prompt(&"日".repeat(9000), &"本".repeat(9000), &issues, &o);
            assert!(
                r.chars().count() <= max,
                "repair {max}: {}",
                r.chars().count()
            );
            let g = generation_prompt(&"日".repeat(9000), 12345, &o);
            assert!(g.chars().count() <= max, "gen {max}: {}", g.chars().count());
        }
    }

    #[test]
    fn prompts_under_the_budget_are_unchanged() {
        let o = options();
        let r = repair_prompt("task", "prev", &["a".to_owned(), "b".to_owned()], &o);
        assert_eq!(
            r,
            "You repair one candidate solution.\n\nTask:\ntask\n\nPrevious candidate:\nprev\n\n\
             The verifier reported these issues:\n- a\n- b\n\n\
             Reply with the corrected candidate only, no commentary."
        );
    }
}
