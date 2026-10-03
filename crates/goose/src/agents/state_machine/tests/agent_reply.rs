//! Covers `Agent::reply_with_state_machine`, the entry point the CLI and desktop
//! reach when the state machine is enabled.

use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::schema::v1::{
    Annotations as AcpAnnotations, ContentBlock as AcpContentBlock, EmbeddedResource,
    EmbeddedResourceResource, ResourceLink, Role as AcpRole, TextContent as AcpTextContent,
    TextResourceContents,
};
use anyhow::Result;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use super::calculator_extension::{value, CalculatorExtension, ADD};
use super::dummy_api::{DummyApi, ProviderFeatures};
use crate::acp::server::GooseAcpAgent;
use crate::agents::extension::ExtensionConfig;
use crate::agents::mcp_client::McpClientTrait;
use crate::agents::subagent_handler::prepare_state_machine_subagent_action_required;
use crate::agents::tool_execution::DECLINED_RESPONSE;
use crate::agents::{Agent, AgentConfig, AgentEvent, GoosePlatform, SessionConfig};
use crate::config::permission::PermissionManager;
use crate::config::GooseMode;
use crate::conversation::message::{ActionRequiredData, Message, MessageContent};
use crate::permission::permission_confirmation::PrincipalType;
use crate::permission::{Permission, PermissionConfirmation};
use crate::providers::base::Provider;
use crate::session::{SessionManager, SessionType};
use goose_providers::model::ModelConfig;

async fn agent_with_dummy_api() -> Result<(Agent, Arc<DummyApi>, String, tempfile::TempDir)> {
    let api = Arc::new(DummyApi::start(ProviderFeatures::default()).await);
    let api_client = goose_providers::api_client::ApiClient::new_with_tls(
        api.uri(),
        goose_providers::api_client::AuthMethod::NoAuth,
        None,
    )?
    .with_request_builder(crate::session_context::session_id_request_builder());
    let provider: Arc<dyn Provider> = Arc::new(
        goose_providers::openai::OpenAiProviderBuilder::new(api_client)
            .name("openai")
            .build(),
    );

    let temp_dir = tempfile::tempdir()?;
    let session_manager = Arc::new(SessionManager::new(temp_dir.path().to_path_buf()));
    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "state-machine-reply".to_string(),
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    let agent = Agent::with_config(AgentConfig::new(
        session_manager,
        Arc::new(PermissionManager::new(temp_dir.path().join("permissions"))),
        None,
        GooseMode::Auto,
        true,
        GoosePlatform::GooseCli,
    ));
    agent
        .update_provider(
            provider,
            ModelConfig::new(goose_providers::openai::OPEN_AI_DEFAULT_MODEL)
                .with_canonical_limits("openai"),
            &session.id,
        )
        .await?;

    Ok((agent, api, session.id, temp_dir))
}

async fn agent_with_calculator() -> Result<(
    Agent,
    Arc<DummyApi>,
    String,
    Arc<CalculatorExtension>,
    tempfile::TempDir,
)> {
    let (agent, api, session_id, temp_dir) = agent_with_dummy_api().await?;
    agent
        .update_goose_mode(GooseMode::Approve, &session_id)
        .await?;
    let calculator = Arc::new(CalculatorExtension::new(
        agent.config.session_manager.action_required(),
    ));
    agent
        .extension_manager
        .add_client(
            "calculator".to_string(),
            ExtensionConfig::Platform {
                name: "calculator".to_string(),
                description: "Stateful test calculator".to_string(),
                display_name: None,
                bundled: None,
                available_tools: vec![],
            },
            calculator.clone(),
            calculator.get_info().cloned(),
        )
        .await;
    Ok((agent, api, session_id, calculator, temp_dir))
}

fn confirmation_ids(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|content| match content {
            MessageContent::ActionRequired(action) => match &action.data {
                ActionRequiredData::ToolConfirmation { id, .. } => Some(id.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

async fn stream_messages(
    mut stream: futures::stream::BoxStream<'_, Result<AgentEvent>>,
) -> Result<Vec<Message>> {
    let mut messages = Vec::new();
    while let Some(event) = stream.next().await {
        if let AgentEvent::Message(message) = event? {
            messages.push(message);
        }
    }
    Ok(messages)
}

#[tokio::test]
async fn state_machine_confirmation_through_agent_resumes_tool_call() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", Some("1"))]);
    let (agent, api, session_id, calculator, _temp_dir) = agent_with_calculator().await?;
    let agent = Arc::new(agent);

    api.on("add one").call(ADD, value(1));
    api.on("result: 1").reply("the result is one");

    let session_config = SessionConfig {
        id: session_id,
        schedule_id: None,
        max_turns: Some(2),
        retry_config: None,
    };
    let mut stream = agent
        .reply(
            Message::user().with_text("add one"),
            session_config.clone(),
            true,
            Some(CancellationToken::new()),
        )
        .await?;
    let mut messages = Vec::new();
    let confirmation_id = loop {
        let event = stream
            .next()
            .await
            .expect("state machine should request confirmation")?;
        if let AgentEvent::Message(message) = event {
            let confirmation_id = confirmation_ids(std::slice::from_ref(&message)).pop();
            messages.push(message);
            if let Some(confirmation_id) = confirmation_id {
                break confirmation_id;
            }
        }
    };
    assert_eq!(calculator.total(), 0);
    {
        let session = agent
            .config
            .session_manager
            .get_session(&session_config.id, true)
            .await?;
        assert!(confirmation_ids(
            session
                .conversation
                .as_ref()
                .expect("session conversation")
                .messages()
        )
        .contains(&confirmation_id));
    }

    agent
        .submit_tool_confirmation(&session_config.id, &confirmation_id, Permission::AllowOnce)
        .await?;
    {
        let session = agent
            .config
            .session_manager
            .get_session(&session_config.id, true)
            .await?;
        assert!(session
            .conversation
            .as_ref()
            .expect("session conversation")
            .messages()
            .iter()
            .any(|message| {
                message.content.iter().any(|content| {
                    matches!(
                        content,
                        MessageContent::ActionRequired(action)
                            if matches!(
                                &action.data,
                                ActionRequiredData::ToolConfirmationResponse { id, permission }
                                    if id == &confirmation_id && permission == &Permission::AllowOnce
                            )
                    )
                })
            }));
    }
    agent
        .submit_tool_confirmation(&session_config.id, &confirmation_id, Permission::AllowOnce)
        .await?;
    assert!(agent
        .submit_tool_confirmation(&session_config.id, &confirmation_id, Permission::DenyOnce)
        .await
        .is_err());
    drop(stream);
    let stream = agent
        .resume_state_machine_turn(session_config.clone(), CancellationToken::new())
        .await?
        .expect("persisted confirmation response should resume the state-machine turn");
    messages.extend(stream_messages(stream).await?);
    assert!(messages.iter().any(|message| message
        .get_tool_response_ids()
        .contains(&confirmation_id.as_str())));
    assert_eq!(calculator.total(), 1);
    assert_eq!(api.call_count(), 2);
    assert!(
        api.calls()
            .iter()
            .all(|call| call.session_id() == Some(session_config.id.as_str())),
        "initial and resumed provider requests must retain the session context"
    );

    assert!(agent
        .submit_tool_confirmation(&session_config.id, &confirmation_id, Permission::AllowOnce)
        .await
        .is_err());
    assert_eq!(calculator.total(), 1);

    assert!(agent
        .submit_tool_confirmation(&session_config.id, "stale-request", Permission::AllowOnce)
        .await
        .is_err());

    let session = agent
        .config
        .session_manager
        .get_session(&session_config.id, true)
        .await?;
    let messages = session
        .conversation
        .as_ref()
        .expect("session conversation")
        .messages();
    let confirmation_responses = messages
        .iter()
        .filter(|message| {
            message.content.iter().any(|content| {
                matches!(
                    content,
                    MessageContent::ActionRequired(action)
                        if matches!(
                            &action.data,
                            ActionRequiredData::ToolConfirmationResponse { id, .. }
                                if id == &confirmation_id
                        )
                )
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(confirmation_responses.len(), 1);
    assert!(!confirmation_responses[0].is_user_visible());
    assert!(!confirmation_responses[0].is_agent_visible());
    assert_eq!(
        messages
            .iter()
            .filter(|message| {
                message.role == rmcp::model::Role::User
                    && message.is_user_visible()
                    && !message.is_tool_response()
            })
            .count(),
        1
    );

    Ok(())
}

#[tokio::test]
async fn probe_legacy_confirmation_does_not_resume_state_machine() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", Some("1"))]);
    let (agent, api, session_id, calculator, _temp_dir) = agent_with_calculator().await?;
    let agent = Arc::new(agent);

    api.on("add one").call(ADD, value(1));
    api.on("result: 1").reply("the result is one");
    let mut stream = agent
        .reply(
            Message::user().with_text("add one"),
            SessionConfig {
                id: session_id.clone(),
                schedule_id: None,
                max_turns: Some(2),
                retry_config: None,
            },
            true,
            Some(CancellationToken::new()),
        )
        .await?;
    let confirmation_id = loop {
        let event = stream.next().await.expect("approval event")?;
        if let AgentEvent::Message(message) = event {
            if let Some(id) = confirmation_ids(&[message]).pop() {
                break id;
            }
        }
    };

    agent
        .handle_confirmation(
            &session_id,
            confirmation_id.clone(),
            PermissionConfirmation {
                principal_type: PrincipalType::Tool,
                permission: Permission::AllowOnce,
            },
        )
        .await;

    let session = agent
        .config
        .session_manager
        .get_session(&session_id, true)
        .await?;
    assert!(session.conversation.as_ref().expect("conversation").messages().iter().all(|message| {
        message.content.iter().all(|content| !matches!(content, MessageContent::ActionRequired(action)
            if matches!(&action.data, ActionRequiredData::ToolConfirmationResponse { id, .. } if id == &confirmation_id)))
    }));
    assert_eq!(calculator.total(), 0);
    assert!(
        tokio::time::timeout(Duration::from_millis(300), stream_messages(stream))
            .await
            .is_err()
    );
    Ok(())
}

#[tokio::test]
async fn probe_forwarded_child_approval_is_unknown_to_parent() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", Some("1"))]);
    let (mut child, api, child_session_id, calculator, _temp_dir) = agent_with_calculator().await?;
    let sessions = child.config.session_manager.clone();
    let parent_session = sessions
        .create_session(
            std::env::temp_dir(),
            "parent".to_string(),
            SessionType::User,
            GooseMode::Approve,
        )
        .await?;
    child.config.confirmation_session_id = Some(parent_session.id.clone());
    let parent = Arc::new(Agent::with_config(
        AgentConfig::new(
            sessions.clone(),
            child.config.permission_manager.clone(),
            None,
            GooseMode::Approve,
            true,
            GoosePlatform::GooseCli,
        )
        .with_tool_confirmation_router(Some(child.tool_confirmation_router.clone())),
    ));
    let child = Arc::new(child);
    assert_eq!(
        sessions
            .get_session(&child_session_id, false)
            .await?
            .goose_mode,
        sessions
            .get_session(&parent_session.id, false)
            .await?
            .goose_mode
    );

    api.on("add one").call(ADD, value(1));
    api.on("result: 1").reply("the result is one");
    let cancel = CancellationToken::new();
    let mut child_stream = child
        .reply(
            Message::user().with_text("add one"),
            SessionConfig {
                id: child_session_id,
                schedule_id: None,
                max_turns: Some(2),
                retry_config: None,
            },
            true,
            Some(cancel.clone()),
        )
        .await?;
    let action_required = loop {
        let event = child_stream.next().await.expect("child approval event")?;
        if let AgentEvent::Message(message) = event {
            if !confirmation_ids(std::slice::from_ref(&message)).is_empty() {
                break message;
            }
        }
    };
    let raw_id = confirmation_ids(std::slice::from_ref(&action_required))
        .pop()
        .unwrap();
    let manager = sessions.action_required();
    let mut parent_approval_stream = manager
        .register_action_required_stream(parent_session.id.clone(), "delegate".to_string())
        .await;
    manager.forward_action_required(&parent_session.id, "delegate", action_required)?;
    let forwarded = tokio::time::timeout(Duration::from_secs(1), parent_approval_stream.recv())
        .await?
        .expect("forwarded approval");
    assert_eq!(confirmation_ids(&[forwarded]), vec![raw_id.clone()]);

    assert!(parent
        .submit_tool_confirmation(&parent_session.id, &raw_id, Permission::AllowOnce)
        .await
        .is_err());
    assert_eq!(calculator.total(), 0);
    assert!(
        tokio::time::timeout(Duration::from_millis(300), stream_messages(child_stream))
            .await
            .is_err()
    );
    cancel.cancel();
    Ok(())
}

async fn relay_fixture() -> Result<(
    Arc<Agent>,
    Arc<Agent>,
    Arc<DummyApi>,
    String,
    String,
    Arc<CalculatorExtension>,
    tempfile::TempDir,
)> {
    let (mut child, api, child_session_id, calculator, temp_dir) = agent_with_calculator().await?;
    let sessions = child.config.session_manager.clone();
    let parent_session = sessions
        .create_session(
            temp_dir.path().to_path_buf(),
            "parent".to_string(),
            SessionType::User,
            GooseMode::Approve,
        )
        .await?;
    child.config.confirmation_session_id = Some(parent_session.id.clone());
    child.config.tool_confirmation_router = Some(child.tool_confirmation_router.clone());
    let parent = Arc::new(Agent::with_config(
        AgentConfig::new(
            sessions,
            child.config.permission_manager.clone(),
            None,
            GooseMode::Approve,
            true,
            GoosePlatform::GooseCli,
        )
        .with_tool_confirmation_router(Some(child.tool_confirmation_router.clone())),
    ));
    Ok((
        parent,
        Arc::new(child),
        api,
        parent_session.id,
        child_session_id,
        calculator,
        temp_dir,
    ))
}

async fn first_confirmation(
    stream: &mut futures::stream::BoxStream<'_, Result<AgentEvent>>,
) -> Result<Message> {
    loop {
        let event = stream.next().await.expect("child approval event")?;
        if let AgentEvent::Message(message) = event {
            if !confirmation_ids(std::slice::from_ref(&message)).is_empty() {
                return Ok(message);
            }
        }
    }
}

async fn relay_approval_case(permission: Permission, expected_total: i64) -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", Some("1"))]);
    let (parent, child, api, parent_session_id, child_session_id, calculator, _temp_dir) =
        relay_fixture().await?;
    api.on("add one")
        .calls([("raw-child-request", ADD, value(1))]);
    api.on("result: 1").reply("the result is one");
    api.on(DECLINED_RESPONSE).reply("the tool was denied");

    let cancel = CancellationToken::new();
    let mut child_stream = child
        .reply(
            Message::user().with_text("add one"),
            SessionConfig {
                id: child_session_id.clone(),
                schedule_id: None,
                max_turns: Some(2),
                retry_config: None,
            },
            true,
            Some(cancel.clone()),
        )
        .await?;
    let original = first_confirmation(&mut child_stream).await?;
    let raw_id = confirmation_ids(std::slice::from_ref(&original))
        .pop()
        .unwrap();
    let forwarded = prepare_state_machine_subagent_action_required(
        &child,
        &child_session_id,
        &original,
        &cancel,
    )
    .await?;
    let display_id = confirmation_ids(std::slice::from_ref(&forwarded))
        .pop()
        .unwrap();
    assert!(display_id.starts_with("summon:"));
    assert_ne!(display_id, raw_id);
    assert_eq!(raw_id, "raw-child-request");

    let action_required = child.config.session_manager.action_required();
    let mut parent_events = action_required
        .register_action_required_stream(parent_session_id.clone(), "delegate".to_string())
        .await;
    action_required.forward_action_required(&parent_session_id, "delegate", forwarded)?;
    let parent_event = tokio::time::timeout(Duration::from_secs(1), parent_events.recv())
        .await?
        .expect("parent action-required event");
    assert_eq!(confirmation_ids(&[parent_event]), vec![display_id.clone()]);

    parent
        .submit_tool_confirmation(&parent_session_id, &display_id, permission.clone())
        .await?;
    let persisted = child
        .config
        .session_manager
        .get_session(&child_session_id, true)
        .await?;
    assert!(persisted
        .conversation
        .as_ref()
        .expect("child conversation")
        .messages()
        .iter()
        .flat_map(|message| &message.content)
        .any(|content| matches!(content, MessageContent::ActionRequired(action)
            if matches!(&action.data, ActionRequiredData::ToolConfirmationResponse { id, permission: recorded }
                if id == &raw_id && recorded == &permission))));

    let result =
        tokio::time::timeout(Duration::from_secs(15), stream_messages(child_stream)).await??;
    assert!(result.iter().any(|message| message.is_tool_response()));
    assert_eq!(calculator.total(), expected_total);
    assert!(parent
        .submit_tool_confirmation(&parent_session_id, &display_id, permission)
        .await
        .is_err());
    assert!(parent
        .submit_tool_confirmation(&parent_session_id, &raw_id, Permission::AllowOnce)
        .await
        .is_err());
    Ok(())
}

#[tokio::test]
async fn state_machine_child_approval_relay_allows_and_resumes() -> Result<()> {
    relay_approval_case(Permission::AllowOnce, 1).await
}

#[tokio::test]
async fn state_machine_child_approval_relay_denies_without_execution() -> Result<()> {
    relay_approval_case(Permission::DenyOnce, 0).await
}

#[tokio::test]
async fn cancelled_state_machine_child_rejects_stale_approval() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", Some("1"))]);
    let (parent, child, api, parent_session_id, child_session_id, calculator, _temp_dir) =
        relay_fixture().await?;
    api.on("add one")
        .calls([("raw-child-request", ADD, value(1))]);
    let cancel = CancellationToken::new();
    let mut child_stream = child
        .reply(
            Message::user().with_text("add one"),
            SessionConfig {
                id: child_session_id.clone(),
                schedule_id: None,
                max_turns: Some(2),
                retry_config: None,
            },
            true,
            Some(cancel.clone()),
        )
        .await?;
    let original = first_confirmation(&mut child_stream).await?;
    let forwarded = prepare_state_machine_subagent_action_required(
        &child,
        &child_session_id,
        &original,
        &cancel,
    )
    .await?;
    let display_id = confirmation_ids(&[forwarded]).pop().unwrap();
    cancel.cancel();
    assert!(parent
        .submit_tool_confirmation(&parent_session_id, &display_id, Permission::AllowOnce)
        .await
        .is_err());
    assert_eq!(calculator.total(), 0);
    assert!(parent
        .submit_tool_confirmation(&parent_session_id, &display_id, Permission::AllowOnce)
        .await
        .is_err());
    drop(child_stream);
    Ok(())
}

#[tokio::test]
async fn concurrent_children_with_reused_raw_ids_route_independently() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", Some("1"))]);
    let (parent, first_child, api, parent_session_id, first_session_id, first_calculator, temp_dir) =
        relay_fixture().await?;
    let sessions = first_child.config.session_manager.clone();
    let second_session = sessions
        .create_session(
            temp_dir.path().to_path_buf(),
            "second child".to_string(),
            SessionType::SubAgent,
            GooseMode::Approve,
        )
        .await?;
    let mut second_config = first_child.config.clone();
    second_config.is_subagent = true;
    let second_child = Arc::new(Agent::with_config(second_config));
    second_child
        .update_provider(
            first_child.provider().await?,
            ModelConfig::new(goose_providers::openai::OPEN_AI_DEFAULT_MODEL)
                .with_canonical_limits("openai"),
            &second_session.id,
        )
        .await?;
    second_child
        .update_goose_mode(GooseMode::Approve, &second_session.id)
        .await?;
    let second_calculator = Arc::new(CalculatorExtension::new(sessions.action_required()));
    second_child
        .extension_manager
        .add_client(
            "calculator".to_string(),
            ExtensionConfig::Platform {
                name: "calculator".to_string(),
                description: "Stateful test calculator".to_string(),
                display_name: None,
                bundled: None,
                available_tools: vec![],
            },
            second_calculator.clone(),
            second_calculator.get_info().cloned(),
        )
        .await;

    api.on("first task")
        .calls([("reused-raw-id", ADD, value(1))]);
    api.on("second task")
        .calls([("reused-raw-id", ADD, value(2))]);
    api.on("result: 1").reply("first complete");
    api.on(DECLINED_RESPONSE).reply("second denied");
    let first_cancel = CancellationToken::new();
    let second_cancel = CancellationToken::new();
    let mut first_stream = first_child
        .reply(
            Message::user().with_text("first task"),
            SessionConfig {
                id: first_session_id.clone(),
                schedule_id: None,
                max_turns: Some(2),
                retry_config: None,
            },
            true,
            Some(first_cancel.clone()),
        )
        .await?;
    let first_original = first_confirmation(&mut first_stream).await?;
    let mut second_stream = second_child
        .reply(
            Message::user().with_text("second task"),
            SessionConfig {
                id: second_session.id.clone(),
                schedule_id: None,
                max_turns: Some(2),
                retry_config: None,
            },
            true,
            Some(second_cancel.clone()),
        )
        .await?;
    let second_original = first_confirmation(&mut second_stream).await?;
    let first_raw = confirmation_ids(std::slice::from_ref(&first_original))
        .pop()
        .unwrap();
    let second_raw = confirmation_ids(std::slice::from_ref(&second_original))
        .pop()
        .unwrap();
    assert_eq!(first_raw, second_raw);
    assert_eq!(first_raw, "reused-raw-id");

    let first_forwarded = prepare_state_machine_subagent_action_required(
        &first_child,
        &first_session_id,
        &first_original,
        &first_cancel,
    )
    .await?;
    let second_forwarded = prepare_state_machine_subagent_action_required(
        &second_child,
        &second_session.id,
        &second_original,
        &second_cancel,
    )
    .await?;
    let first_display = confirmation_ids(&[first_forwarded]).pop().unwrap();
    let second_display = confirmation_ids(&[second_forwarded]).pop().unwrap();
    assert_ne!(first_display, second_display);
    assert!(parent
        .submit_tool_confirmation(&first_session_id, &second_display, Permission::AllowOnce)
        .await
        .is_err());

    parent
        .submit_tool_confirmation(&parent_session_id, &second_display, Permission::DenyOnce)
        .await?;
    parent
        .submit_tool_confirmation(&parent_session_id, &first_display, Permission::AllowOnce)
        .await?;
    tokio::time::timeout(Duration::from_secs(15), stream_messages(first_stream)).await??;
    tokio::time::timeout(Duration::from_secs(15), stream_messages(second_stream)).await??;
    assert_eq!(first_calculator.total(), 1);
    assert_eq!(second_calculator.total(), 0);
    Ok(())
}

#[tokio::test]
async fn reply_streams_the_turn_and_ends() -> Result<()> {
    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    api.on("are you there?").reply("still here");

    let session_config = SessionConfig {
        id: session_id.clone(),
        schedule_id: None,
        max_turns: Some(2),
        retry_config: None,
    };
    let stream = agent
        .reply_with_state_machine(
            Message::user().with_text("are you there?"),
            session_config,
            Some(CancellationToken::new()),
        )
        .await?;

    let replies = tokio::time::timeout(Duration::from_secs(30), async move {
        tokio::pin!(stream);
        let mut replies = Vec::new();
        while let Some(event) = stream.next().await {
            if let AgentEvent::Message(message) = event? {
                replies.push(message.as_concat_text());
            }
        }
        anyhow::Ok(replies)
    })
    .await??;

    assert!(
        replies.iter().any(|reply| reply == "still here"),
        "expected the scripted reply, got {replies:?}"
    );
    assert_eq!(api.call_count(), 1);

    Ok(())
}

#[tokio::test]
async fn bang_shell_uses_state_machine_when_explicitly_enabled() -> Result<()> {
    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    let session_config = SessionConfig {
        id: session_id,
        schedule_id: None,
        max_turns: Some(2),
        retry_config: None,
    };
    let stream = agent
        .reply(
            Message::user().with_text("!echo hello"),
            session_config,
            true,
            Some(CancellationToken::new()),
        )
        .await?;
    tokio::pin!(stream);
    let mut requested_shell = false;
    while let Some(event) = stream.next().await {
        if let AgentEvent::Message(message) = event? {
            requested_shell |= message.content.iter().any(|content| {
                matches!(
                    content,
                    crate::conversation::message::MessageContent::ToolRequest(request)
                        if request.tool_call.as_ref().is_ok_and(|call| call.name == "shell")
                )
            });
        }
    }

    assert!(requested_shell);
    assert_eq!(api.call_count(), 0);

    Ok(())
}

async fn reply_messages(
    agent: &Agent,
    session_id: String,
    message: Message,
) -> Result<Vec<Message>> {
    let stream = agent
        .reply(
            message,
            SessionConfig {
                id: session_id,
                schedule_id: None,
                max_turns: Some(2),
                retry_config: None,
            },
            crate::agents::state_machine::enabled(),
            Some(CancellationToken::new()),
        )
        .await?;
    tokio::pin!(stream);
    let mut messages = Vec::new();
    while let Some(event) = stream.next().await {
        if let AgentEvent::Message(message) = event? {
            messages.push(message);
        }
    }
    Ok(messages)
}

fn assistant_only_acp_annotations() -> AcpAnnotations {
    AcpAnnotations::new().audience(vec![AcpRole::Assistant])
}

fn assistant_only_acp_text(text: &str) -> AcpContentBlock {
    AcpContentBlock::Text(AcpTextContent::new(text).annotations(assistant_only_acp_annotations()))
}

fn empty_audience_acp_annotations() -> AcpAnnotations {
    AcpAnnotations::new().audience(Vec::new())
}

fn empty_audience_acp_text(text: &str) -> AcpContentBlock {
    AcpContentBlock::Text(AcpTextContent::new(text).annotations(empty_audience_acp_annotations()))
}

fn assistant_only_embedded_resource(text: &str) -> AcpContentBlock {
    AcpContentBlock::Resource(
        EmbeddedResource::new(EmbeddedResourceResource::TextResourceContents(
            TextResourceContents::new(text, "file:///hidden-resource.txt"),
        ))
        .annotations(assistant_only_acp_annotations()),
    )
}

fn empty_audience_embedded_resource(text: &str) -> AcpContentBlock {
    AcpContentBlock::Resource(
        EmbeddedResource::new(EmbeddedResourceResource::TextResourceContents(
            TextResourceContents::new(text, "file:///empty-audience-resource.txt"),
        ))
        .annotations(empty_audience_acp_annotations()),
    )
}

fn assistant_only_resource_link(text: &str) -> Result<(AcpContentBlock, tempfile::NamedTempFile)> {
    let file = tempfile::NamedTempFile::new()?;
    std::fs::write(file.path(), text)?;
    let uri = url::Url::from_file_path(file.path())
        .map_err(|()| anyhow::anyhow!("temporary resource path is not a valid file URL"))?;
    let link = ResourceLink::new("hidden-resource.txt", uri.to_string())
        .annotations(assistant_only_acp_annotations());
    Ok((AcpContentBlock::ResourceLink(link), file))
}

fn shell_commands(messages: &[Message]) -> Vec<&str> {
    messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|content| match content {
            MessageContent::ToolRequest(request) => request
                .tool_call
                .as_ref()
                .ok()
                .filter(|call| call.name == "shell")
                .and_then(|call| call.arguments.as_ref())
                .and_then(|arguments| arguments.get("command"))
                .and_then(serde_json::Value::as_str),
            _ => None,
        })
        .collect()
}

async fn assert_bang_shell_uses_only_user_visible_content() -> Result<()> {
    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    api.on("benign visible input")
        .reply("handled as ordinary input");
    let hidden_text_prefix = GooseAcpAgent::convert_acp_prompt_to_message(&[
        assistant_only_acp_text("!echo hidden"),
        AcpContentBlock::Text(AcpTextContent::new("benign visible input")),
    ]);
    let messages = reply_messages(&agent, session_id, hidden_text_prefix).await?;
    assert!(shell_commands(&messages).is_empty());
    assert_eq!(api.call_count(), 1);

    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    api.on("benign visible input")
        .reply("handled as ordinary input");
    let empty_audience_text = GooseAcpAgent::convert_acp_prompt_to_message(&[
        empty_audience_acp_text("!echo hidden"),
        AcpContentBlock::Text(AcpTextContent::new("benign visible input")),
    ]);
    let messages = reply_messages(&agent, session_id, empty_audience_text).await?;
    assert!(shell_commands(&messages).is_empty());
    assert_eq!(api.call_count(), 1);

    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    let hidden_text_suffix = GooseAcpAgent::convert_acp_prompt_to_message(&[
        AcpContentBlock::Text(AcpTextContent::new("!echo visible")),
        assistant_only_acp_text("&& echo hidden"),
    ]);
    let messages = reply_messages(&agent, session_id, hidden_text_suffix).await?;
    assert_eq!(shell_commands(&messages), ["echo visible"]);
    assert_eq!(api.call_count(), 0);

    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    api.on("benign visible input")
        .reply("handled as ordinary input");
    let hidden_resource_prefix = GooseAcpAgent::convert_acp_prompt_to_message(&[
        assistant_only_embedded_resource("!echo hidden"),
        AcpContentBlock::Text(AcpTextContent::new("benign visible input")),
    ]);
    let messages = reply_messages(&agent, session_id, hidden_resource_prefix).await?;
    assert!(shell_commands(&messages).is_empty());
    assert_eq!(api.call_count(), 1);

    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    api.on("benign visible input")
        .reply("handled as ordinary input");
    let empty_audience_resource = GooseAcpAgent::convert_acp_prompt_to_message(&[
        empty_audience_embedded_resource("!echo hidden"),
        AcpContentBlock::Text(AcpTextContent::new("benign visible input")),
    ]);
    let messages = reply_messages(&agent, session_id, empty_audience_resource).await?;
    assert!(shell_commands(&messages).is_empty());
    assert_eq!(api.call_count(), 1);

    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    let (hidden_link, _resource_file) = assistant_only_resource_link("&& echo hidden")?;
    let hidden_link_suffix = GooseAcpAgent::convert_acp_prompt_to_message(&[
        AcpContentBlock::Text(AcpTextContent::new("!echo visible")),
        hidden_link,
    ]);
    let messages = reply_messages(&agent, session_id, hidden_link_suffix).await?;
    assert_eq!(shell_commands(&messages), ["echo visible"]);
    assert_eq!(api.call_count(), 0);

    Ok(())
}

#[tokio::test]
async fn bang_shell_not_executed_in_legacy_loop() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", None::<&str>)]);
    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    api.on("!echo hello").reply("treated as text");
    let messages =
        reply_messages(&agent, session_id, Message::user().with_text("!echo hello")).await?;
    assert!(shell_commands(&messages).is_empty());
    assert_eq!(api.call_count(), 1);
    Ok(())
}

#[tokio::test]
async fn bang_shell_visibility_is_enforced_when_state_machine_is_enabled() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", Some("1"))]);
    assert_bang_shell_uses_only_user_visible_content().await
}
