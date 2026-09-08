use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::StreamExt;
use goose::agents::extension::ExtensionConfig;
use goose::agents::mcp_client::{Error as McpError, McpClientTrait};
use goose::agents::{
    Agent, AgentConfig, AgentEvent, GoosePlatform, SessionConfig, ToolCallContext,
};
use goose::config::permission::{PermissionLevel, PermissionManager};
use goose::config::GooseMode;
use goose::conversation::message::{ActionRequiredData, Message, MessageContent};
use goose::permission::permission_confirmation::PrincipalType;
use goose::permission::{Permission, PermissionConfirmation};
use goose::providers::base::{stream_from_single_message, MessageStream, Provider};
use goose::session::{SessionManager, SessionType};
use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ContentBlock, InitializeResult, JsonObject,
    ListToolsResult, Tool,
};
use rmcp::object;
use test_case::test_case;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const WAIT: &str = "ordered_wait";
const AUTO: &str = "ordered_auto";
const ALIAS: &str = "developer__ordered_wait";
const PARALLEL: &str = "parallel";
const TEST_TIMEOUT: Duration = Duration::from_secs(30);

struct ScriptedProvider {
    replies: Mutex<VecDeque<Message>>,
}

#[async_trait]
impl Provider for ScriptedProvider {
    fn get_name(&self) -> &str {
        "ordered-tools-test"
    }

    async fn stream(
        &self,
        _model_config: &ModelConfig,
        _system: &str,
        _messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let message = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("agent requested more responses than the test scripted");
        Ok(stream_from_single_message(
            message,
            ProviderUsage::new("ordered-tools-test".into(), Usage::default()),
        ))
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Notice {
    Confirmation(String),
    Entered(String),
    Exited(String),
    AwaitingResume,
}

struct NativeClient {
    notices: mpsc::UnboundedSender<Notice>,
    entries: Mutex<Vec<String>>,
    releases: Mutex<HashMap<String, CancellationToken>>,
}

struct InvocationGuard {
    notices: mpsc::UnboundedSender<Notice>,
    id: String,
}

impl Drop for InvocationGuard {
    fn drop(&mut self) {
        let _ = self.notices.send(Notice::Exited(self.id.clone()));
    }
}

#[async_trait]
impl McpClientTrait for NativeClient {
    fn get_info(&self) -> Option<&InitializeResult> {
        None
    }

    async fn list_tools(
        &self,
        _session_id: &str,
        _next_cursor: Option<String>,
        _cancel_token: CancellationToken,
    ) -> Result<ListToolsResult, McpError> {
        // Advertising the alias keeps the test about scheduling. The state
        // machine separately rejects names absent from the advertised catalog.
        Ok(ListToolsResult::with_all_items(
            [WAIT, AUTO, ALIAS, PARALLEL]
                .into_iter()
                .map(|name| {
                    Tool::new(
                        name.to_string(),
                        "Record invocation order for the integration test",
                        object!({
                            "type": "object",
                            "properties": {"block": {"type": "boolean"}}
                        }),
                    )
                })
                .collect(),
        ))
    }

    async fn call_tool(
        &self,
        ctx: &ToolCallContext,
        name: &str,
        arguments: Option<JsonObject>,
        cancel_token: CancellationToken,
    ) -> Result<CallToolResult, McpError> {
        assert!([WAIT, AUTO, PARALLEL].contains(&name));
        let id = ctx.tool_call_request_id.clone().unwrap();
        let release = CancellationToken::new();
        self.releases
            .lock()
            .unwrap()
            .insert(id.clone(), release.clone());
        self.entries.lock().unwrap().push(id.clone());
        let _guard = InvocationGuard {
            notices: self.notices.clone(),
            id: id.clone(),
        };
        self.notices.send(Notice::Entered(id.clone())).unwrap();
        if arguments
            .as_ref()
            .and_then(|args| args.get("block"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            tokio::select! {
                _ = release.cancelled() => {},
                _ = cancel_token.cancelled() => {
                    return Ok(CallToolResult::error(vec![ContentBlock::text("cancelled")]));
                }
            }
        }
        Ok(CallToolResult::success(vec![ContentBlock::text(id)]))
    }
}

struct Harness {
    agent: Arc<Agent>,
    provider: Arc<ScriptedProvider>,
    session_id: String,
    client: Arc<NativeClient>,
    notices: mpsc::UnboundedReceiver<Notice>,
    confirmation_ids: Arc<Mutex<HashMap<String, String>>>,
    state_machine: bool,
    resume_confirmations: Mutex<Option<mpsc::UnboundedSender<Message>>>,
    _temp_dir: tempfile::TempDir,
}

struct RunningReply(JoinHandle<Result<Vec<Message>>>);

impl Drop for RunningReply {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl RunningReply {
    async fn finish(&mut self) -> Result<Vec<Message>> {
        tokio::time::timeout(TEST_TIMEOUT, &mut self.0).await??
    }

    async fn drop_stream(&mut self) {
        self.0.abort();
        let error = (&mut self.0).await.unwrap_err();
        assert!(error.is_cancelled());
    }
}

fn call(id: &str, name: &str, block: bool) -> (String, CallToolRequestParams) {
    (
        id.to_string(),
        CallToolRequestParams::new(name.to_string()).with_arguments(object!({"block": block})),
    )
}

impl Harness {
    async fn new(
        calls: Vec<(String, CallToolRequestParams)>,
        bound: usize,
        wait_permission: PermissionLevel,
    ) -> Result<Self> {
        let temp_dir = tempfile::tempdir()?;
        let session_manager = Arc::new(SessionManager::new(temp_dir.path().join("sessions")));
        let permission_manager =
            Arc::new(PermissionManager::new(temp_dir.path().join("permissions")));
        permission_manager.update_user_permission(WAIT, wait_permission);
        for name in [AUTO, ALIAS, PARALLEL] {
            permission_manager.update_user_permission(name, PermissionLevel::AlwaysAllow);
        }
        let session = session_manager
            .create_session(
                temp_dir.path().to_path_buf(),
                "ordered tools".into(),
                SessionType::Hidden,
                GooseMode::Approve,
            )
            .await?;
        let mut config = AgentConfig::new(
            session_manager,
            permission_manager,
            None,
            GooseMode::Approve,
            true,
            GoosePlatform::GooseCli,
        )
        .with_use_login_shell_path(false)
        .with_ordered_tool_calls(
            [WAIT, AUTO, ALIAS].into_iter().map(str::to_string),
            NonZeroUsize::new(bound).unwrap(),
        );
        // Isolate the test from plugins installed in the developer's account.
        config.is_subagent = true;
        let agent = Arc::new(Agent::with_config(config));
        let mut request = Message::assistant();
        for (id, tool_call) in calls {
            request = request.with_tool_request(id, Ok(tool_call));
        }
        let provider = Arc::new(ScriptedProvider {
            replies: Mutex::new(VecDeque::from([
                request,
                Message::assistant().with_text("finished"),
            ])),
        });
        agent
            .update_provider(
                provider.clone(),
                ModelConfig::new("test-model"),
                &session.id,
            )
            .await?;
        let (notices_tx, notices) = mpsc::unbounded_channel();
        let client = Arc::new(NativeClient {
            notices: notices_tx,
            entries: Mutex::new(Vec::new()),
            releases: Mutex::new(HashMap::new()),
        });
        agent
            .extension_manager
            .add_ephemeral_client(
                "developer".into(),
                ExtensionConfig::Builtin {
                    name: "developer".into(),
                    description: "native scheduling test client".into(),
                    display_name: None,
                    timeout: None,
                    bundled: None,
                    available_tools: Vec::new(),
                },
                client.clone(),
                None,
                None,
            )
            .await;
        Ok(Self {
            agent,
            provider,
            session_id: session.id,
            client,
            notices,
            confirmation_ids: Arc::new(Mutex::new(HashMap::new())),
            state_machine: goose::agents::state_machine::enabled(),
            resume_confirmations: Mutex::new(None),
            _temp_dir: temp_dir,
        })
    }

    fn start(&self, cancel: CancellationToken) -> RunningReply {
        let agent = self.agent.clone();
        let session_id = self.session_id.clone();
        let notices = self.client.notices.clone();
        let confirmation_ids = self.confirmation_ids.clone();
        let state_machine = self.state_machine;
        let (resume_tx, mut resume_rx) = mpsc::unbounded_channel();
        *self.resume_confirmations.lock().unwrap() = Some(resume_tx);
        RunningReply(tokio::spawn(async move {
            let mut messages = Vec::new();
            let mut input = Message::user().with_text("run the scripted tools");
            let mut pending_confirmations = HashSet::new();
            loop {
                let mut stream = agent
                    .reply(
                        input,
                        SessionConfig {
                            id: session_id.clone(),
                            schedule_id: None,
                            max_turns: Some(3),
                            retry_config: None,
                        },
                        Some(cancel.clone()),
                    )
                    .await?;
                while let Some(event) = stream.next().await {
                    if let AgentEvent::Message(message) = event? {
                        for content in &message.content {
                            if let MessageContent::ActionRequired(action) = content {
                                if let ActionRequiredData::ToolConfirmation { id, .. } =
                                    &action.data
                                {
                                    let request_id = id
                                        .strip_prefix(&format!("{session_id}:"))
                                        .unwrap_or(id)
                                        .to_string();
                                    confirmation_ids
                                        .lock()
                                        .unwrap()
                                        .insert(request_id.clone(), id.clone());
                                    let _ = notices.send(Notice::Confirmation(request_id));
                                    if state_machine {
                                        pending_confirmations.insert(id.clone());
                                    }
                                }
                            }
                        }
                        messages.push(message);
                    }
                }
                if pending_confirmations.is_empty() {
                    break;
                }
                // The state machine yields at approval. Its next reply persists
                // the hidden confirmation response and resumes the same turn.
                let _ = notices.send(Notice::AwaitingResume);
                input = tokio::select! {
                    _ = cancel.cancelled() => break,
                    response = resume_rx.recv() => response.ok_or_else(|| anyhow!("confirmation channel closed"))?,
                };
                for content in &input.content {
                    if let MessageContent::ActionRequired(action) = content {
                        if let ActionRequiredData::ToolConfirmationResponse { id, .. } =
                            &action.data
                        {
                            pending_confirmations.remove(id);
                        }
                    }
                }
            }
            Ok(messages)
        }))
    }

    fn entries(&self) -> Vec<String> {
        self.client.entries.lock().unwrap().clone()
    }

    fn release(&self, id: &str) {
        self.client
            .releases
            .lock()
            .unwrap()
            .get(id)
            .unwrap()
            .cancel();
    }

    async fn notice(&mut self, expected: Notice) -> Result<()> {
        tokio::time::timeout(TEST_TIMEOUT, async {
            while let Some(notice) = self.notices.recv().await {
                if notice == expected {
                    return Ok(());
                }
            }
            Err(anyhow!("notice channel closed before {expected:?}"))
        })
        .await?
    }

    async fn entered(&mut self, ids: &[&str]) -> Result<()> {
        tokio::time::timeout(TEST_TIMEOUT, async {
            while !ids
                .iter()
                .all(|id| self.entries().iter().any(|entry| entry == id))
            {
                self.notices
                    .recv()
                    .await
                    .ok_or_else(|| anyhow!("notice channel closed"))?;
            }
            Ok(())
        })
        .await?
    }

    async fn confirm(&self, id: &str, permission: Permission) {
        let confirmation_id = self
            .confirmation_ids
            .lock()
            .unwrap()
            .get(id)
            .expect("confirmation must be observed before responding")
            .clone();
        if self.state_machine {
            self.resume_confirmations
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .send(
                    Message::user()
                        .with_content(MessageContent::action_required_tool_confirmation_response(
                            confirmation_id,
                            permission,
                        ))
                        .with_visibility(false, false),
                )
                .unwrap();
            return;
        }
        self.agent
            .handle_confirmation(
                confirmation_id,
                PermissionConfirmation {
                    principal_type: PrincipalType::Tool,
                    permission,
                },
            )
            .await;
    }
}

fn assert_error_response(messages: &[Message], id: &str) {
    let response = messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|content| match content {
            MessageContent::ToolResponse(response) if response.id == id => Some(response),
            _ => None,
        })
        .next_back()
        .expect("expected a paired tool response");
    assert!(
        match &response.tool_result {
            Ok(result) => result.is_error == Some(true),
            Err(_) => true,
        },
        "expected an error response for {id}: {response:?}"
    );
}

#[test_case(false; "legacy")]
#[test_case(true; "state_machine")]
#[tokio::test]
async fn source_order_survives_mixed_permission_buckets(state_machine: bool) -> Result<()> {
    let _env = env_lock::lock_env([(
        "GOOSE_STATE_MACHINE",
        Some(if state_machine { "1" } else { "0" }),
    )]);
    let mut harness = Harness::new(
        vec![call("first", WAIT, true), call("second", AUTO, false)],
        8,
        PermissionLevel::AskBefore,
    )
    .await?;
    let mut run = harness.start(CancellationToken::new());
    harness.notice(Notice::Confirmation("first".into())).await?;
    harness.confirm("first", Permission::AllowOnce).await;
    harness.entered(&["first"]).await?;
    assert_eq!(harness.entries(), ["first"]);
    harness.release("first");
    run.finish().await?;
    assert_eq!(harness.entries(), ["first", "second"]);
    Ok(())
}

#[test_case(false; "legacy")]
#[test_case(true; "state_machine")]
#[tokio::test]
async fn denied_first_call_does_not_block_following_calls(state_machine: bool) -> Result<()> {
    let _env = env_lock::lock_env([(
        "GOOSE_STATE_MACHINE",
        Some(if state_machine { "1" } else { "0" }),
    )]);
    let mut harness = Harness::new(
        vec![call("denied", WAIT, false), call("allowed", AUTO, false)],
        8,
        PermissionLevel::AskBefore,
    )
    .await?;
    let mut run = harness.start(CancellationToken::new());
    harness
        .notice(Notice::Confirmation("denied".into()))
        .await?;
    harness.confirm("denied", Permission::DenyOnce).await;
    let messages = run.finish().await?;
    assert_eq!(harness.entries(), ["allowed"]);
    assert_error_response(&messages, "denied");
    Ok(())
}

#[test_case(false; "legacy")]
#[test_case(true; "state_machine")]
#[tokio::test]
async fn inspection_denial_does_not_consume_execution_or_block_the_lane(
    state_machine: bool,
) -> Result<()> {
    let _env = env_lock::lock_env([(
        "GOOSE_STATE_MACHINE",
        Some(if state_machine { "1" } else { "0" }),
    )]);
    let harness = Harness::new(
        vec![call("denied", WAIT, false), call("allowed", AUTO, false)],
        8,
        PermissionLevel::NeverAllow,
    )
    .await?;
    let mut run = harness.start(CancellationToken::new());
    let messages = run.finish().await?;
    assert_eq!(harness.entries(), ["allowed"]);
    assert_error_response(&messages, "denied");
    Ok(())
}

#[test_case(false; "legacy")]
#[test_case(true; "state_machine")]
#[tokio::test]
async fn unrelated_tools_run_while_ordered_call_is_blocked(state_machine: bool) -> Result<()> {
    let _env = env_lock::lock_env([(
        "GOOSE_STATE_MACHINE",
        Some(if state_machine { "1" } else { "0" }),
    )]);
    let mut harness = Harness::new(
        vec![
            call("first", AUTO, true),
            call("second", AUTO, false),
            call("unrelated", PARALLEL, false),
        ],
        8,
        PermissionLevel::AlwaysAllow,
    )
    .await?;
    let mut run = harness.start(CancellationToken::new());
    harness.entered(&["first", "unrelated"]).await?;
    assert_eq!(harness.entries().len(), 2);
    harness.release("first");
    run.finish().await?;
    assert_eq!(
        harness
            .entries()
            .iter()
            .filter(|id| id.as_str() != "unrelated")
            .cloned()
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    Ok(())
}

#[test_case(false; "legacy")]
#[test_case(true; "state_machine")]
#[tokio::test]
async fn overflow_never_invokes_the_rejected_body(state_machine: bool) -> Result<()> {
    let _env = env_lock::lock_env([(
        "GOOSE_STATE_MACHINE",
        Some(if state_machine { "1" } else { "0" }),
    )]);
    let harness = Harness::new(
        vec![
            call("first", AUTO, false),
            call("second", AUTO, false),
            call("overflow", AUTO, false),
            call("unrelated", PARALLEL, false),
        ],
        2,
        PermissionLevel::AlwaysAllow,
    )
    .await?;
    let mut run = harness.start(CancellationToken::new());
    let messages = run.finish().await?;
    let entries = harness.entries();
    assert_eq!(entries.len(), 3);
    assert!(!entries.iter().any(|id| id == "overflow"));
    assert_error_response(&messages, "overflow");
    Ok(())
}

#[test_case(false, false; "legacy_stop")]
#[test_case(true, false; "state_machine_stop")]
#[test_case(false, true; "legacy_stream_drop")]
#[test_case(true, true; "state_machine_stream_drop")]
#[tokio::test]
async fn cancellation_and_stream_drop_never_invoke_waiting_calls(
    state_machine: bool,
    drop_stream: bool,
) -> Result<()> {
    let _env = env_lock::lock_env([(
        "GOOSE_STATE_MACHINE",
        Some(if state_machine { "1" } else { "0" }),
    )]);
    let mut harness = Harness::new(
        vec![call("first", AUTO, true), call("waiting", AUTO, false)],
        8,
        PermissionLevel::AlwaysAllow,
    )
    .await?;
    let cancel = CancellationToken::new();
    let mut run = harness.start(cancel.clone());
    harness.entered(&["first"]).await?;
    if drop_stream {
        run.drop_stream().await;
    } else {
        cancel.cancel();
        run.finish().await?;
    }
    harness.notice(Notice::Exited("first".into())).await?;
    assert_eq!(harness.entries(), ["first"]);
    Ok(())
}

#[test_case(false; "legacy")]
#[test_case(true; "state_machine")]
#[tokio::test]
async fn configured_aliases_share_the_ordered_lane(state_machine: bool) -> Result<()> {
    let _env = env_lock::lock_env([(
        "GOOSE_STATE_MACHINE",
        Some(if state_machine { "1" } else { "0" }),
    )]);
    let mut harness = Harness::new(
        vec![call("canonical", WAIT, true), call("alias", ALIAS, false)],
        8,
        PermissionLevel::AlwaysAllow,
    )
    .await?;
    let mut run = harness.start(CancellationToken::new());
    harness.entered(&["canonical"]).await?;
    assert_eq!(harness.entries(), ["canonical"]);
    harness.release("canonical");
    run.finish().await?;
    assert_eq!(harness.entries(), ["canonical", "alias"]);
    Ok(())
}

#[tokio::test]
async fn state_machine_runs_unrelated_tools_then_yields_for_ordered_approval() -> Result<()> {
    let _env = env_lock::lock_env([("GOOSE_STATE_MACHINE", Some("1"))]);
    let mut harness = Harness::new(
        vec![
            call("first", WAIT, false),
            call("second", AUTO, false),
            call("unrelated", PARALLEL, false),
        ],
        8,
        PermissionLevel::AskBefore,
    )
    .await?;
    let mut run = harness.start(CancellationToken::new());
    // Legacy resolves its approval phase before polling any tool streams. This
    // check covers the state machine's existing independently executable tools.
    harness.notice(Notice::AwaitingResume).await?;
    assert_eq!(harness.entries(), ["unrelated"]);
    assert_eq!(harness.provider.replies.lock().unwrap().len(), 1);
    harness.confirm("first", Permission::AllowOnce).await;
    run.finish().await?;
    assert_eq!(harness.entries(), ["unrelated", "first", "second"]);
    assert!(harness.provider.replies.lock().unwrap().is_empty());
    Ok(())
}

#[test_case(false; "legacy")]
#[test_case(true; "state_machine")]
#[tokio::test]
async fn approval_resume_preserves_the_original_ordered_batch_bound(
    state_machine: bool,
) -> Result<()> {
    let _env = env_lock::lock_env([(
        "GOOSE_STATE_MACHINE",
        Some(if state_machine { "1" } else { "0" }),
    )]);
    let mut harness = Harness::new(
        vec![
            call("first", AUTO, false),
            call("second", WAIT, false),
            call("overflow", AUTO, false),
        ],
        2,
        PermissionLevel::AskBefore,
    )
    .await?;
    let mut run = harness.start(CancellationToken::new());
    harness
        .notice(Notice::Confirmation("second".into()))
        .await?;
    harness.confirm("second", Permission::AllowOnce).await;
    let messages = run.finish().await?;
    assert_eq!(harness.entries(), ["first", "second"]);
    assert_error_response(&messages, "overflow");
    Ok(())
}

#[test_case(false; "legacy")]
#[test_case(true; "state_machine")]
#[tokio::test]
async fn partial_approval_keeps_the_group_and_original_bound_intact(
    state_machine: bool,
) -> Result<()> {
    let _env = env_lock::lock_env([(
        "GOOSE_STATE_MACHINE",
        Some(if state_machine { "1" } else { "0" }),
    )]);
    let mut harness = Harness::new(
        vec![
            call("head", AUTO, false),
            call("approval_a", WAIT, false),
            call("approval_b", WAIT, false),
        ],
        2,
        PermissionLevel::AskBefore,
    )
    .await?;
    let mut run = harness.start(CancellationToken::new());
    if state_machine {
        harness.notice(Notice::AwaitingResume).await?;
    } else {
        harness
            .notice(Notice::Confirmation("approval_a".into()))
            .await?;
    }
    harness.confirm("approval_a", Permission::AllowOnce).await;
    if state_machine {
        harness.notice(Notice::AwaitingResume).await?;
        assert_eq!(harness.provider.replies.lock().unwrap().len(), 1);
    } else {
        harness
            .notice(Notice::Confirmation("approval_b".into()))
            .await?;
    }
    assert!(harness.entries().is_empty());
    harness.confirm("approval_b", Permission::AllowOnce).await;
    let messages = run.finish().await?;
    assert_eq!(harness.entries(), ["head", "approval_a"]);
    assert_error_response(&messages, "approval_b");
    Ok(())
}
