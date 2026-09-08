use async_stream::try_stream;
use futures::stream::{self, BoxStream};
use futures::{Stream, StreamExt};
use rmcp::model::CallToolResult;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use std::path::PathBuf;

use crate::config::permission::PermissionLevel;
use crate::conversation::message::Message;
use crate::mcp_utils::ToolResult;
use crate::permission::Permission;
use rmcp::model::{ContentBlock, ServerNotification};

#[derive(Clone)]
pub(crate) struct ToolCallNotificationEmitter {
    sender: mpsc::Sender<ServerNotification>,
}

impl ToolCallNotificationEmitter {
    pub(crate) fn new(sender: mpsc::Sender<ServerNotification>) -> Self {
        Self { sender }
    }

    pub(crate) fn emit_best_effort(&self, notification: ServerNotification) {
        // Do not let a slow notification consumer delay tool execution.
        let _ = self.sender.try_send(notification);
    }
}

/// Context passed through the tool call dispatch chain.
#[derive(Clone)]
pub struct ToolCallContext {
    pub session_id: String,
    pub working_dir: Option<PathBuf>,
    pub tool_call_request_id: Option<String>,
    notification_emitter: Option<ToolCallNotificationEmitter>,
}

impl ToolCallContext {
    pub fn new(
        session_id: String,
        working_dir: Option<PathBuf>,
        tool_call_request_id: Option<String>,
    ) -> Self {
        Self {
            session_id,
            working_dir,
            tool_call_request_id,
            notification_emitter: None,
        }
    }

    pub fn working_dir_str(&self) -> Option<&str> {
        self.working_dir.as_ref().and_then(|p| p.to_str())
    }

    pub(crate) fn with_notification_emitter(
        mut self,
        notification_emitter: ToolCallNotificationEmitter,
    ) -> Self {
        self.notification_emitter = Some(notification_emitter);
        self
    }

    pub(crate) fn notification_emitter(&self) -> Option<&ToolCallNotificationEmitter> {
        self.notification_emitter.as_ref()
    }
}

// ToolCallResult combines the result of a tool call with an optional notification stream that
// can be used to receive notifications from the tool.
pub struct ToolCallResult {
    pub result: Box<dyn Future<Output = ToolResult<rmcp::model::CallToolResult>> + Send + Unpin>,
    pub notification_stream: Option<Box<dyn Stream<Item = ServerNotification> + Send + Unpin>>,
    pub action_required_stream: Option<Box<dyn Stream<Item = Message> + Send + Unpin>>,
}

impl From<ToolResult<rmcp::model::CallToolResult>> for ToolCallResult {
    fn from(result: ToolResult<rmcp::model::CallToolResult>) -> Self {
        Self {
            result: Box::new(futures::future::ready(result)),
            notification_stream: None,
            action_required_stream: None,
        }
    }
}

use crate::agents::Agent;
use crate::conversation::message::ToolRequest;
use crate::session::Session;
use crate::tool_inspection::get_security_finding_id_from_results;

pub(super) enum ToolStreamItem<T> {
    ActionRequired(Message),
    Message(ServerNotification),
    Result(T),
}

pub(super) type ToolStream =
    Pin<Box<dyn Stream<Item = ToolStreamItem<ToolResult<CallToolResult>>> + Send>>;

/// Host-selected tools that share one sequential execution lane per tool batch.
/// Names are matched exactly; include every supported alias in the same group.
#[derive(Clone, Debug)]
pub struct OrderedToolCalls {
    tool_names: HashSet<String>,
    max_calls: NonZeroUsize,
}

impl OrderedToolCalls {
    pub(super) fn new(
        tool_names: impl IntoIterator<Item = String>,
        max_calls: NonZeroUsize,
    ) -> Self {
        Self {
            tool_names: tool_names.into_iter().collect(),
            max_calls,
        }
    }

    pub(super) fn contains(&self, tool_name: &str) -> bool {
        self.tool_names.contains(tool_name)
    }
}

type IdentifiedToolStream =
    BoxStream<'static, (String, ToolStreamItem<ToolResult<CallToolResult>>)>;

fn identify_stream(request_id: String, stream: ToolStream) -> IdentifiedToolStream {
    stream.map(move |item| (request_id.clone(), item)).boxed()
}

/// Schedule only streams that survived permission checks. Missing denied or
/// malformed requests never reserve a lane position. The lane holds cold
/// streams, so dropping the batch cannot start its queued tool bodies.
pub(super) fn schedule_tool_streams(
    tool_streams: Vec<(String, ToolStream)>,
    requests: &[ToolRequest],
    policy: Option<&OrderedToolCalls>,
    cancel: CancellationToken,
) -> IdentifiedToolStream {
    let Some(policy) = policy else {
        return stream::select_all(
            tool_streams
                .into_iter()
                .map(|(request_id, stream)| identify_stream(request_id, stream)),
        )
        .boxed();
    };

    // Request metadata can originate at the provider. Scheduling policy comes
    // solely from the host configuration and the original request order.
    let positions: HashMap<&str, usize> = requests
        .iter()
        .enumerate()
        .filter(|(_, request)| {
            request
                .tool_call
                .as_ref()
                .is_ok_and(|call| policy.contains(call.name.as_ref()))
        })
        .map(|(position, request)| (request.id.as_str(), position))
        .collect();

    let mut concurrent = Vec::new();
    let mut ordered = Vec::new();
    for (request_id, stream) in tool_streams {
        if let Some(position) = positions.get(request_id.as_str()) {
            ordered.push((*position, request_id, stream));
        } else {
            concurrent.push(identify_stream(request_id, stream));
        }
    }
    ordered.sort_by_key(|(position, _, _)| *position);

    let mut admitted = Vec::new();
    for (index, (_, request_id, stream)) in ordered.into_iter().enumerate() {
        if index < policy.max_calls.get() {
            admitted.push(identify_stream(request_id, stream));
        } else {
            drop(stream);
            let result = CallToolResult::error(vec![ContentBlock::text(format!(
                "This tool call was not executed because the ordered tool batch exceeds its limit of {} calls. Wait for the admitted calls to finish, then submit this work in a later batch.",
                policy.max_calls,
            ))]);
            concurrent.push(
                stream::once(futures::future::ready((
                    request_id,
                    ToolStreamItem::Result(Ok(result)),
                )))
                .boxed(),
            );
        }
    }
    if !admitted.is_empty() {
        concurrent.push(
            stream::iter(admitted)
                .take_while(move |_| futures::future::ready(!cancel.is_cancelled()))
                .flatten()
                .boxed(),
        );
    }
    stream::select_all(concurrent).boxed()
}

#[cfg(test)]
mod ordered_tool_tests {
    use super::*;
    use futures::FutureExt;
    use rmcp::model::CallToolRequestParams;
    use std::sync::{Arc, Mutex};
    use tokio::sync::oneshot;

    fn policy(max_calls: usize) -> OrderedToolCalls {
        OrderedToolCalls::new(
            ["python".to_string(), "developer__python".to_string()],
            NonZeroUsize::new(max_calls).unwrap(),
        )
    }

    fn request(id: &str, name: &str) -> ToolRequest {
        ToolRequest {
            id: id.to_string(),
            tool_call: Ok(CallToolRequestParams::new(name.to_string())),
            metadata: None,
            tool_meta: None,
        }
    }

    fn call(
        id: &str,
        starts: &Arc<Mutex<Vec<String>>>,
        wait: Option<oneshot::Receiver<()>>,
    ) -> (String, ToolStream) {
        let id = id.to_string();
        let body_id = id.clone();
        let starts = starts.clone();
        let body = async move {
            starts.lock().unwrap().push(body_id.clone());
            if let Some(wait) = wait {
                let _ = wait.await;
            }
            Ok(CallToolResult::success(vec![ContentBlock::text(body_id)]))
        };
        (id, tool_stream(stream::empty(), stream::empty(), body))
    }

    async fn next_result(stream: &mut IdentifiedToolStream) -> (String, CallToolResult) {
        let (id, item) = stream.next().await.expect("expected a tool result");
        let ToolStreamItem::Result(result) = item else {
            panic!("expected a result item");
        };
        (id, result.unwrap())
    }

    #[tokio::test]
    async fn ordered_calls_follow_request_order_across_aliases() {
        let starts = Arc::new(Mutex::new(Vec::new()));
        let requests = [
            request("first", "python"),
            request("second", "developer__python"),
        ];
        let mut scheduled = schedule_tool_streams(
            vec![call("second", &starts, None), call("first", &starts, None)],
            &requests,
            Some(&policy(2)),
            CancellationToken::new(),
        );
        assert_eq!(next_result(&mut scheduled).await.0, "first");
        assert_eq!(*starts.lock().unwrap(), ["first"]);
        assert_eq!(next_result(&mut scheduled).await.0, "second");
        assert_eq!(*starts.lock().unwrap(), ["first", "second"]);
        assert!(scheduled.next().await.is_none());
    }

    #[tokio::test]
    async fn denied_and_malformed_requests_leave_no_queue_holes() {
        let starts = Arc::new(Mutex::new(Vec::new()));
        let mut malformed = request("malformed", "python");
        malformed.tool_call = Err(rmcp::model::ErrorData::invalid_params("invalid call", None));
        let requests = [
            request("denied", "python"),
            malformed,
            request("approved", "python"),
        ];
        let mut scheduled = schedule_tool_streams(
            vec![call("approved", &starts, None)],
            &requests,
            Some(&policy(1)),
            CancellationToken::new(),
        );
        assert_eq!(next_result(&mut scheduled).await.0, "approved");
        assert_eq!(*starts.lock().unwrap(), ["approved"]);
    }

    #[tokio::test]
    async fn unrelated_tools_run_while_the_ordered_lane_is_waiting() {
        let starts = Arc::new(Mutex::new(Vec::new()));
        let (release, wait) = oneshot::channel();
        let requests = [
            request("first", "python"),
            request("second", "python"),
            request("other", "read"),
        ];
        let mut scheduled = schedule_tool_streams(
            vec![
                call("second", &starts, None),
                call("first", &starts, Some(wait)),
                call("other", &starts, None),
            ],
            &requests,
            Some(&policy(2)),
            CancellationToken::new(),
        );
        assert_eq!(next_result(&mut scheduled).await.0, "other");
        assert!(scheduled.next().now_or_never().is_none());
        assert!(!starts.lock().unwrap().contains(&"second".to_string()));
        release.send(()).unwrap();
        assert_eq!(next_result(&mut scheduled).await.0, "first");
        assert_eq!(next_result(&mut scheduled).await.0, "second");
    }

    #[tokio::test]
    async fn ordered_overflow_never_polls_the_rejected_tool() {
        let starts = Arc::new(Mutex::new(Vec::new()));
        let requests = [request("first", "python"), request("overflow", "python")];
        let mut scheduled = schedule_tool_streams(
            vec![
                call("overflow", &starts, None),
                call("first", &starts, None),
            ],
            &requests,
            Some(&policy(1)),
            CancellationToken::new(),
        );
        let mut results = HashMap::new();
        for _ in 0..2 {
            let (id, result) = next_result(&mut scheduled).await;
            results.insert(id, result);
        }
        assert_eq!(results["overflow"].is_error, Some(true));
        assert_ne!(results["first"].is_error, Some(true));
        assert_eq!(*starts.lock().unwrap(), ["first"]);
    }

    #[tokio::test]
    async fn dropping_a_batch_never_polls_its_waiting_calls() {
        let starts = Arc::new(Mutex::new(Vec::new()));
        let (_release, wait) = oneshot::channel();
        let (queued_release, queued_wait) = oneshot::channel();
        let requests = [request("first", "python"), request("queued", "python")];
        let mut scheduled = schedule_tool_streams(
            vec![
                call("first", &starts, Some(wait)),
                call("queued", &starts, Some(queued_wait)),
            ],
            &requests,
            Some(&policy(2)),
            CancellationToken::new(),
        );
        assert!(scheduled.next().now_or_never().is_none());
        assert_eq!(*starts.lock().unwrap(), ["first"]);
        drop(scheduled);
        assert!(queued_release.is_closed());
        assert_eq!(*starts.lock().unwrap(), ["first"]);
    }

    #[tokio::test]
    async fn cancellation_does_not_advance_to_the_next_ordered_call() {
        let starts = Arc::new(Mutex::new(Vec::new()));
        let requests = [request("first", "python"), request("queued", "python")];
        let cancel = CancellationToken::new();
        let mut scheduled = schedule_tool_streams(
            vec![call("first", &starts, None), call("queued", &starts, None)],
            &requests,
            Some(&policy(2)),
            cancel.clone(),
        );
        assert_eq!(next_result(&mut scheduled).await.0, "first");
        cancel.cancel();
        assert!(scheduled.next().await.is_none());
        assert_eq!(*starts.lock().unwrap(), ["first"]);
    }

    #[tokio::test]
    async fn separate_batches_do_not_share_admission_or_a_lock() {
        let starts = Arc::new(Mutex::new(Vec::new()));
        let (_release, wait) = oneshot::channel();
        let configuration = policy(1);
        let requests = [request("same-id", "python")];
        let mut first = schedule_tool_streams(
            vec![call("same-id", &starts, Some(wait))],
            &requests,
            Some(&configuration),
            CancellationToken::new(),
        );
        assert!(first.next().now_or_never().is_none());
        let mut second = schedule_tool_streams(
            vec![call("same-id", &starts, None)],
            &requests,
            Some(&configuration),
            CancellationToken::new(),
        );
        assert_eq!(next_result(&mut second).await.0, "same-id");
        assert_eq!(*starts.lock().unwrap(), ["same-id", "same-id"]);
    }

    #[tokio::test]
    async fn default_scheduling_keeps_tools_concurrent() {
        let starts = Arc::new(Mutex::new(Vec::new()));
        let (_release, wait) = oneshot::channel();
        let requests = [request("first", "python"), request("second", "python")];
        let mut scheduled = schedule_tool_streams(
            vec![
                call("first", &starts, Some(wait)),
                call("second", &starts, None),
            ],
            &requests,
            None,
            CancellationToken::new(),
        );
        assert_eq!(next_result(&mut scheduled).await.0, "second");
        assert_eq!(*starts.lock().unwrap(), ["first", "second"]);
    }

    #[tokio::test]
    async fn provider_metadata_cannot_place_a_tool_in_the_ordered_lane() {
        let starts = Arc::new(Mutex::new(Vec::new()));
        let (_release, wait) = oneshot::channel();
        let mut unrelated = request("other", "read");
        unrelated.tool_meta = Some(serde_json::json!({
            "goose.execution_mode": "sequential",
            "goose_extension": "developer",
            "tool_name": "python"
        }));
        let requests = [request("first", "python"), unrelated];
        let mut scheduled = schedule_tool_streams(
            vec![
                call("first", &starts, Some(wait)),
                call("other", &starts, None),
            ],
            &requests,
            Some(&policy(1)),
            CancellationToken::new(),
        );
        assert_eq!(next_result(&mut scheduled).await.0, "other");
    }
}

pub(super) fn tool_stream<S, A, F>(rx: S, action_required_rx: A, done: F) -> ToolStream
where
    S: Stream<Item = ServerNotification> + Send + Unpin + 'static,
    A: Stream<Item = Message> + Send + Unpin + 'static,
    F: Future<Output = ToolResult<CallToolResult>> + Send + 'static,
{
    Box::pin(async_stream::stream! {
        tokio::pin!(done);
        let mut rx = rx;
        let mut action_required_rx = action_required_rx;

        loop {
            tokio::select! {
                Some(msg) = action_required_rx.next() => {
                    yield ToolStreamItem::ActionRequired(msg);
                }
                Some(msg) = rx.next() => {
                    yield ToolStreamItem::Message(msg);
                }
                r = &mut done => {
                    yield ToolStreamItem::Result(r);
                    break;
                }
            }
        }
    })
}

pub const DECLINED_RESPONSE: &str = "The user has declined to run this tool. \
    DO NOT attempt to call this tool again. \
    If there are no alternative methods to proceed, clearly explain the situation and STOP.";

pub const CHAT_MODE_TOOL_SKIPPED_RESPONSE: &str = "Let the user know the tool call was skipped in goose chat mode. \
                                        DO NOT apologize for skipping the tool call. DO NOT say sorry. \
                                        Provide an explanation of what the tool call would do, structured as a \
                                        plan for the user. Again, DO NOT apologize. \
                                        **Example Plan:**\n \
                                        1. **Identify Task Scope** - Determine the purpose and expected outcome.\n \
                                        2. **Outline Steps** - Break down the steps.\n \
                                        If needed, adjust the explanation based on user preferences or questions.";

impl Agent {
    pub(super) fn handle_approval_tool_requests<'a>(
        &'a self,
        tool_requests: &'a [ToolRequest],
        tool_futures: &'a mut Vec<(String, ToolStream)>,
        request_to_response_map: &'a mut HashMap<String, Message>,
        cancellation_token: Option<CancellationToken>,
        session: &'a Session,
        inspection_results: &'a [crate::tool_inspection::InspectionResult],
    ) -> BoxStream<'a, anyhow::Result<Message>> {
        try_stream! {
        for request in tool_requests.iter() {
            if let Ok(tool_call) = request.tool_call.clone() {
                let security_message = inspection_results.iter()
                    .find(|result| result.tool_request_id == request.id)
                    .and_then(|result| {
                        if let crate::tool_inspection::InspectionAction::RequireApproval(Some(message)) = &result.action {
                            Some(message.clone())
                        } else {
                            None
                        }
                    });

                // Subagent confirmations share the parent's
                // ToolConfirmationRouter, so keys are namespaced by the
                // subagent session. Some providers use per-conversation
                // sequential ids, and a subagent id could otherwise collide
                // with a pending parent id in the shared router.
                let confirmation_id = if self.config.is_subagent {
                    format!("{}:{}", session.id, request.id)
                } else {
                    request.id.clone()
                };

                let confirmation_rx = self
                    .tool_confirmation_router
                    .register(confirmation_id.clone())
                    .await;

                let action_required_msg = Message::assistant()
                    .with_action_required(
                        confirmation_id,
                        tool_call.name.to_string().clone(),
                        tool_call.arguments.clone().unwrap_or_default(),
                        security_message,
                    )
                    .user_only();
                yield action_required_msg;

                let confirmation = confirmation_rx.await
                    .map_err(|_| anyhow::anyhow!("Confirmation channel closed for request {}", request.id))?;

                if let Some(finding_id) = get_security_finding_id_from_results(&request.id, inspection_results) {
                    let action = match confirmation.permission {
                        Permission::AllowOnce | Permission::AlwaysAllow => "ALLOW",
                        _ => "BLOCK",
                    };
                    tracing::info!(
                        monotonic_counter.goose.prompt_injection_user_decisions = 1,
                        security.event_type = "user_decision",
                        security.action = action,
                        security.finding_id = %finding_id,
                        tool.request_id = %request.id,
                        user.decision = ?confirmation.permission,
                        "security finding: user decision"
                    );
                }

                if confirmation.permission == Permission::AllowOnce || confirmation.permission == Permission::AlwaysAllow {
                    let (req_id, tool_result) = self.dispatch_tool_call(tool_call.clone(), request.id.clone(), cancellation_token.clone(), session).await;

                    tool_futures.push((req_id, match tool_result {
                        Ok(result) => tool_stream(
                            result.notification_stream.unwrap_or_else(|| Box::new(stream::empty())),
                            result.action_required_stream.unwrap_or_else(|| Box::new(stream::empty())),
                            result.result,
                        ),
                        Err(e) => tool_stream(
                            Box::new(stream::empty()),
                            Box::new(stream::empty()),
                            futures::future::ready(Err(e)),
                        ),
                    }));

                    if confirmation.permission == Permission::AlwaysAllow {
                        self.tool_inspection_manager
                            .update_permission_manager(&tool_call.name, PermissionLevel::AlwaysAllow)
                            .await;
                    }
                } else {
                    if let Some(response) = request_to_response_map.get_mut(&request.id) {
                        response.add_tool_response_with_metadata(
                            request.id.clone(),
                            Ok(CallToolResult::error(vec![ContentBlock::text(DECLINED_RESPONSE)])),
                            request.metadata.as_ref(),
                        );
                    }

                    if confirmation.permission == Permission::AlwaysDeny {
                        self.tool_inspection_manager
                            .update_permission_manager(&tool_call.name, PermissionLevel::NeverAllow)
                            .await;
                    }
                }
            }
        }
    }.boxed()
    }

    pub(crate) fn handle_frontend_tool_request<'a>(
        &'a self,
        tool_request: &'a ToolRequest,
        message_tool_response: &'a mut Message,
    ) -> BoxStream<'a, anyhow::Result<Message>> {
        try_stream! {
                if let Ok(tool_call) = tool_request.tool_call.clone() {
                    if self.is_frontend_tool(&tool_call.name).await {
                        yield Message::assistant().with_frontend_tool_request(
                            tool_request.id.clone(),
                            Ok(tool_call.clone())
                        );

                        if let Some((id, result)) = self.tool_result_rx.lock().await.recv().await {
                            message_tool_response.add_tool_response_with_metadata(
                                id,
                                result,
                                tool_request.metadata.as_ref(),
                            );
                        }
                    }
            }
        }
        .boxed()
    }
}
