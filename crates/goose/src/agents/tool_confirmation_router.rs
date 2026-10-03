use std::collections::HashMap;
use std::sync::{Arc, Weak};

use anyhow::{anyhow, Result};
use tokio::sync::{oneshot, Mutex};
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::agents::Agent;
use crate::permission::PermissionConfirmation;

enum PendingConfirmation {
    Direct(oneshot::Sender<PermissionConfirmation>),
    StateMachineSubagent {
        agent: Weak<Agent>,
        session_id: String,
        request_id: String,
        cancel: CancellationToken,
    },
}

impl PendingConfirmation {
    fn is_active(&self) -> bool {
        match self {
            Self::Direct(sender) => !sender.is_closed(),
            Self::StateMachineSubagent { agent, cancel, .. } => {
                !cancel.is_cancelled() && agent.strong_count() > 0
            }
        }
    }
}

type PendingConfirmations = HashMap<(String, String), PendingConfirmation>;

/// Routes permission confirmations to the awaiting tool request.
///
/// Clones share the same pending map, which lets subagents register their
/// confirmations in the parent's router so confirmations delivered to the
/// parent agent reach the subagent that is waiting for them.
#[derive(Clone)]
pub struct ToolConfirmationRouter {
    pending: Arc<Mutex<PendingConfirmations>>,
}

impl Default for ToolConfirmationRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolConfirmationRouter {
    pub(super) fn new() -> Self {
        Self {
            pending: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub(super) async fn register(
        &self,
        session_id: String,
        request_id: String,
    ) -> oneshot::Receiver<PermissionConfirmation> {
        let (tx, rx) = oneshot::channel();
        let mut pending = self.pending.lock().await;
        pending.retain(|_, route| route.is_active());
        pending.insert((session_id, request_id), PendingConfirmation::Direct(tx));
        rx
    }

    pub(super) async fn register_state_machine_subagent(
        &self,
        parent_session_id: String,
        display_request_id: String,
        agent: Weak<Agent>,
        child_session_id: String,
        child_request_id: String,
        cancel: CancellationToken,
    ) -> Result<()> {
        let mut pending = self.pending.lock().await;
        pending.retain(|_, route| route.is_active());
        let key = (parent_session_id, display_request_id);
        if pending.contains_key(&key) {
            return Err(anyhow!("duplicate subagent confirmation route"));
        }
        pending.insert(
            key,
            PendingConfirmation::StateMachineSubagent {
                agent,
                session_id: child_session_id,
                request_id: child_request_id,
                cancel,
            },
        );
        Ok(())
    }

    pub(super) async fn deliver(
        &self,
        session_id: &str,
        request_id: &str,
        confirmation: PermissionConfirmation,
    ) -> bool {
        match self
            .deliver_with_ack(session_id, request_id, confirmation)
            .await
        {
            Ok(delivered) => delivered,
            Err(error) => {
                warn!(request_id = %request_id, "Confirmation delivery failed: {error}");
                false
            }
        }
    }

    /// A relayed state-machine decision is acknowledged only after the child
    /// agent has persisted it to its own session.
    pub(super) async fn deliver_with_ack(
        &self,
        session_id: &str,
        request_id: &str,
        confirmation: PermissionConfirmation,
    ) -> Result<bool> {
        let key = (session_id.to_string(), request_id.to_string());
        let route = self.pending.lock().await.remove(&key);
        match route {
            Some(PendingConfirmation::Direct(tx)) => {
                if tx.send(confirmation).is_err() {
                    warn!(
                        request_id = %request_id,
                        "Confirmation receiver was dropped (task cancelled)"
                    );
                    Ok(false)
                } else {
                    Ok(true)
                }
            }
            Some(PendingConfirmation::StateMachineSubagent {
                agent,
                session_id,
                request_id,
                cancel,
            }) => {
                if cancel.is_cancelled() {
                    return Err(anyhow!("subagent confirmation request was cancelled"));
                }
                let agent = agent
                    .upgrade()
                    .ok_or_else(|| anyhow!("subagent confirmation request is no longer active"))?;
                Box::pin(agent.submit_tool_confirmation(
                    &session_id,
                    &request_id,
                    confirmation.permission,
                ))
                .await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permission::permission_confirmation::PrincipalType;
    use crate::permission::Permission;

    fn test_confirmation() -> PermissionConfirmation {
        PermissionConfirmation {
            principal_type: PrincipalType::Tool,
            permission: Permission::AllowOnce,
        }
    }

    #[tokio::test]
    async fn test_register_then_deliver() {
        let router = ToolConfirmationRouter::new();
        let rx = router
            .register("session_1".to_string(), "req_1".to_string())
            .await;
        assert!(
            router
                .deliver("session_1", "req_1", test_confirmation())
                .await
        );
        let confirmation = rx.await.unwrap();
        assert_eq!(confirmation.permission, Permission::AllowOnce);
    }

    #[tokio::test]
    async fn cloned_router_delivers_subagent_confirmation_once() {
        let parent_router = ToolConfirmationRouter::new();
        let subagent_router = parent_router.clone();
        let rx = subagent_router
            .register("parent".to_string(), "subagent:request".to_string())
            .await;

        assert!(
            parent_router
                .deliver("parent", "subagent:request", test_confirmation())
                .await
        );
        assert_eq!(rx.await.unwrap().permission, Permission::AllowOnce);
        assert!(
            !parent_router
                .deliver("parent", "subagent:request", test_confirmation())
                .await
        );
    }

    #[tokio::test]
    async fn test_deliver_unknown_request() {
        let router = ToolConfirmationRouter::new();
        assert!(
            !router
                .deliver("session_1", "unknown", test_confirmation())
                .await
        );
    }

    #[tokio::test]
    async fn test_request_cannot_be_delivered_from_another_session() {
        let router = ToolConfirmationRouter::new();
        let _rx = router
            .register("session_1".to_string(), "req_1".to_string())
            .await;

        assert!(
            !router
                .deliver("session_2", "req_1", test_confirmation())
                .await
        );
        assert_eq!(router.pending.lock().await.len(), 1);
    }

    #[tokio::test]
    async fn test_cancelled_receiver() {
        let router = ToolConfirmationRouter::new();
        let rx = router
            .register("session_1".to_string(), "req_1".to_string())
            .await;
        drop(rx); // simulate task cancellation
        assert!(
            !router
                .deliver("session_1", "req_1", test_confirmation())
                .await
        );
    }

    #[tokio::test]
    async fn test_stale_entries_pruned_on_register() {
        let router = ToolConfirmationRouter::new();
        let rx = router
            .register("session_1".to_string(), "req_1".to_string())
            .await;
        drop(rx); // simulate task cancellation — entry is now stale

        assert_eq!(router.pending.lock().await.len(), 1);

        let _rx2 = router
            .register("session_1".to_string(), "req_2".to_string())
            .await;
        assert_eq!(router.pending.lock().await.len(), 1); // only req_2 remains
        assert!(router
            .pending
            .lock()
            .await
            .contains_key(&("session_1".to_string(), "req_2".to_string())));
    }

    #[tokio::test]
    async fn test_concurrent_requests_out_of_order() {
        use std::sync::Arc;

        let router = Arc::new(ToolConfirmationRouter::new());

        // Register two requests
        let rx1 = router
            .register("session_1".to_string(), "req_1".to_string())
            .await;
        let rx2 = router
            .register("session_1".to_string(), "req_2".to_string())
            .await;

        // Deliver in reverse order
        assert!(
            router
                .deliver(
                    "session_1",
                    "req_2",
                    PermissionConfirmation {
                        principal_type: PrincipalType::Tool,
                        permission: Permission::DenyOnce,
                    }
                )
                .await
        );
        assert_eq!(router.pending.lock().await.len(), 1);
        assert!(
            router
                .deliver("session_1", "req_1", test_confirmation())
                .await
        );
        assert_eq!(router.pending.lock().await.len(), 0);

        let c1 = rx1.await.unwrap();
        assert_eq!(c1.permission, Permission::AllowOnce);
        let c2 = rx2.await.unwrap();
        assert_eq!(c2.permission, Permission::DenyOnce);
    }
}
