//! The `doc-router` binary's local subcommands, driven as a real process.

mod common;

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use common::{fixture_path, scratch_dir};
use httpmock::prelude::*;
use serde_json::Value;

fn doc_router() -> Command {
    Command::cargo_bin("doc-router").expect("the doc-router binary is built for this test")
}

/// Run the CLI, assert it exited 0, and parse its stdout as JSON.
fn json_output(args: &[&str]) -> Value {
    let output = doc_router().args(args).output().expect("the CLI runs");
    assert!(
        output.status.success(),
        "`doc-router {}` failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap_or_else(|e| {
        panic!(
            "`doc-router {}` did not print JSON ({e}): {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

fn mixed() -> String {
    fixture_path("mixed.pdf").display().to_string()
}

#[test]
fn classify_json_reports_the_pages_that_need_ocr() {
    let value = json_output(&["classify", "--json", &mixed()]);

    assert_eq!(value["pdf_type"], "mixed");
    assert_eq!(value["page_count"], 4);
    assert_eq!(value["pages_needing_ocr"], serde_json::json!([1, 3]));
    assert_eq!(value["is_complex_layout"], false);
}

#[test]
fn plan_json_prints_the_decision_and_its_metadata() {
    let value = json_output(&["plan", "--json", &mixed()]);

    let route = &value["decision"]["route"];
    assert_eq!(route["plan"]["reason"], "mixed_split");
    assert_eq!(route["plan"]["routed_model"], "mistral-ocr");
    assert_eq!(route["classification"]["page_count"], 4);

    let legs = route["plan"]["legs"].as_array().expect("two legs");
    assert_eq!(legs.len(), 2);
    assert_eq!(legs[0]["model"], "local_pdf/extract");
    assert_eq!(legs[0]["pages"], serde_json::json!([0, 2]));
    assert_eq!(legs[1]["model"], "mistral-ocr");
    assert_eq!(legs[1]["pages"], serde_json::json!([1, 3]));

    let metadata = &value["metadata"];
    assert_eq!(metadata["reason"], "mixed_split");
    assert_eq!(metadata["tier"], "standard");
    assert_eq!(metadata["page_count"], 4);
    assert_eq!(metadata["pages_needing_ocr"], serde_json::json!([1, 3]));
    assert_eq!(metadata["split"], true);
}

#[test]
fn plan_honours_default_model_and_exits_zero_on_a_bypass() {
    let not_a_pdf = fixture_path("not_a_pdf.bin").display().to_string();
    let value = json_output(&[
        "plan",
        "--json",
        &not_a_pdf,
        "--default-model",
        "gpt-4o-mini",
    ]);

    assert_eq!(value["decision"]["bypass"]["reason"], "not_pdf");
    assert_eq!(value["metadata"]["tier"], "bypass");
    assert_eq!(value["metadata"]["reason"], "not_pdf");
    assert_eq!(value["metadata"]["routed_model"], "gpt-4o-mini");
}

#[test]
fn split_writes_a_pdf_that_classifies_to_the_kept_pages() {
    let dir = scratch_dir("cli-split");
    let out = dir.join("subset.pdf");

    let value = json_output(&[
        "split",
        "--json",
        &mixed(),
        "--pages",
        "1,3",
        "--out",
        &out.display().to_string(),
    ]);
    assert_eq!(value["pages"], serde_json::json!([1, 3]));
    assert!(value["bytes"].as_u64().unwrap_or(0) > 0);

    let bytes = std::fs::read(&out).expect("the subset PDF was written");
    let classification = doc_router::classify(&bytes).expect("the subset is a readable PDF");
    assert_eq!(classification.page_count, 2);

    // And the CLI agrees with the library about it.
    let via_cli = json_output(&["classify", "--json", &out.display().to_string()]);
    assert_eq!(via_cli["page_count"], 2);
}

#[test]
fn extract_writes_one_markdown_file_per_page() {
    let dir = scratch_dir("cli-extract");

    let value = json_output(&[
        "extract",
        "--json",
        &mixed(),
        "--pages",
        "0,2",
        "--out",
        &dir.display().to_string(),
    ]);

    let files = value["files"].as_array().expect("a file list");
    assert_eq!(files.len(), 2);
    for file in files {
        let path = Path::new(file.as_str().expect("a path"));
        assert!(path.exists(), "{} was not written", path.display());
    }
    assert!(dir.join("page-0.md").exists());
    assert!(dir.join("page-2.md").exists());
}

#[test]
fn a_bad_config_file_fails_loudly() {
    let dir = scratch_dir("cli-bad-config");
    let config = dir.join("config.json");
    std::fs::write(&config, r#"{"tiers": {"local": "", "standard": ""}}"#).expect("write config");

    doc_router()
        .args([
            "--config",
            &config.display().to_string(),
            "classify",
            &mixed(),
        ])
        .assert()
        .failure();
}

/// Both hosted-judge variables, removed rather than blanked, so a checkout with
/// a real key in the environment still exercises the no-key path here. Nothing
/// in this file may reach the Jev API: a test that costs money is a test nobody
/// runs.
fn without_a_hosted_key() -> Command {
    let mut command = doc_router();
    command.env_remove("TYPESAFE_API_KEY");
    command.env_remove("JEV_API_KEY");
    command
}

/// `--judge <name>`'s help, for one subcommand.
fn help_for(subcommand: &str) -> String {
    let output = doc_router()
        .args([subcommand, "--help"])
        .output()
        .expect("the CLI runs");
    assert!(output.status.success(), "`{subcommand} --help` failed");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Run the CLI expecting it to fail, and return what it said on stderr.
fn failure_stderr(command: &mut Command, args: &[&str]) -> String {
    let output = command.args(args).output().expect("the CLI runs");
    assert!(
        !output.status.success(),
        "`doc-router {}` was expected to fail; stdout: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stdout)
    );
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn the_default_judge_is_the_built_in_heuristic() {
    for args in [
        vec!["classify", "--json", &mixed()],
        vec!["plan", "--json", &mixed()],
    ] {
        let value = json_output(&args);
        assert_eq!(value["judge"]["name"], "heuristic", "{args:?}");
        // The heuristic answers every page itself, so there is nothing to
        // report and the key is absent rather than an empty list.
        assert!(value["judge"]["fallbacks"].is_null(), "{args:?}");
    }
}

/// Naming the default judge is the same run as not naming one — the flag
/// selects a judge, it does not switch on a different code path.
#[test]
fn asking_for_the_default_judge_by_name_changes_nothing() {
    let implicit = json_output(&["classify", "--json", &mixed()]);
    let explicit = json_output(&["classify", "--json", "--judge", "heuristic", &mixed()]);

    assert_eq!(implicit["pages_needing_ocr"], explicit["pages_needing_ocr"]);
    assert_eq!(implicit["judge"], explicit["judge"]);
}

/// The default judge is also invisible in the human-readable output: every line
/// the CLI printed before this flag existed still reads the same.
#[test]
fn the_default_judge_is_not_announced_in_the_human_summary() {
    let output = doc_router()
        .args(["classify", &mixed()])
        .output()
        .expect("the CLI runs");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "{stdout}");
    assert_eq!(stdout.lines().count(), 1, "{stdout}");
    assert!(!stdout.contains("judge"), "{stdout}");
    assert!(output.stderr.is_empty(), "{:?}", output.stderr);
}

#[test]
fn an_unknown_judge_is_an_error_that_lists_the_registered_names() {
    let stderr = failure_stderr(
        &mut doc_router(),
        &["classify", "--judge", "heuristik", &mixed()],
    );

    assert!(stderr.contains("unknown judge `heuristik`"), "{stderr}");
    for name in ["heuristic", "jev", "jev_gated"] {
        assert!(stderr.contains(name), "{name} missing from: {stderr}");
    }
}

/// A registered judge with no credentials is a different failure from a name
/// nobody registered, and says so: the fix is a variable, not a spelling.
#[test]
fn a_hosted_judge_without_a_key_names_the_variables_and_does_not_fall_back() {
    for subcommand in ["classify", "plan"] {
        let stderr = failure_stderr(
            &mut without_a_hosted_key(),
            &[subcommand, "--judge", "jev", &mixed()],
        );

        assert!(stderr.contains("judge `jev` is unavailable"), "{stderr}");
        assert!(stderr.contains("TYPESAFE_API_KEY"), "{stderr}");
        assert!(stderr.contains("JEV_API_KEY"), "{stderr}");
        // Not a typo, and not a run that quietly used the heuristic instead.
        assert!(!stderr.contains("unknown judge"), "{stderr}");
        assert!(!stderr.contains("heuristic"), "{stderr}");
    }
}

/// `run` refuses before it opens a socket: discovering a typo after a paid OCR
/// call would be an expensive way to learn about it.
#[test]
fn run_rejects_an_unusable_judge_before_it_calls_the_proxy() {
    let stderr = failure_stderr(
        &mut without_a_hosted_key(),
        &[
            "run",
            "--judge",
            "jev",
            "--base-url",
            // Unroutable on purpose: reaching it would be the bug.
            "http://127.0.0.1:1",
            &mixed(),
        ],
    );

    assert!(stderr.contains("judge `jev` is unavailable"), "{stderr}");
}

#[test]
fn the_judge_help_lists_every_registered_name() {
    for subcommand in ["classify", "plan", "run"] {
        let help = help_for(subcommand);
        assert!(help.contains("--judge <NAME>"), "{subcommand}: {help}");
        for name in ["heuristic", "jev", "jev_gated"] {
            assert!(
                help.contains(name),
                "{subcommand} help lacks {name}: {help}"
            );
        }
        assert!(
            help.contains("(default: heuristic)"),
            "{subcommand}: {help}"
        );
    }
}

/// The flag is only on the subcommands where a judge changes the answer.
#[test]
fn the_subcommands_that_ask_no_judge_have_no_judge_flag() {
    for subcommand in ["extract", "split"] {
        let help = help_for(subcommand);
        assert!(!help.contains("--judge"), "{subcommand}: {help}");
    }
}

/// A hosted judge whose API is down, end to end, against a local mock: the
/// document still routes — the right default for one document an operator is
/// waiting on — and the CLI says out loud that the verdicts are not the ones
/// that were asked for. The key here is a string, not a credential: nothing in
/// this test leaves the machine.
#[test]
fn a_hosted_judge_that_cannot_answer_falls_back_visibly() {
    let server = MockServer::start();
    let down = server.mock(|when, then| {
        when.any_request();
        then.status(503).body("service unavailable");
    });

    let output = doc_router()
        .env("TYPESAFE_API_KEY", "sk-not-a-real-key")
        .env("TYPESAFE_BASE_URL", server.base_url())
        .args(["classify", "--json", "--judge", "jev", &mixed()])
        .output()
        .expect("the CLI runs");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "{stderr}");
    assert!(down.calls() >= 1, "the judge never called its API");

    let value: Value = serde_json::from_str(&stdout).expect("the CLI printed JSON");
    // The heuristic's answer, because that is what the fallback produces.
    assert_eq!(value["pages_needing_ocr"], serde_json::json!([1, 3]));
    // Under the name that was asked for, with the fallback recorded rather than
    // swallowed.
    assert_eq!(value["judge"]["name"], "jev");
    let fallbacks = value["judge"]["fallbacks"]
        .as_array()
        .unwrap_or_else(|| panic!("a fallback should be recorded: {stdout}"));
    assert!(!fallbacks.is_empty(), "{stdout}");
    assert!(
        fallbacks.iter().all(|f| f["reason"]
            .as_str()
            .is_some_and(|r| r.starts_with("jev_fallback_"))),
        "{stdout}"
    );
    assert!(stderr.contains("fell back to the heuristic"), "{stderr}");
}
