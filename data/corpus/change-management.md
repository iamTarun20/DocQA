# Production change management

A Northwind Labs production change needs an owner, a rollout checklist and a rollback decision. File the change in Signalboard with the affected services and the check that will indicate the rollout is healthy. Include a reference to the code review or infrastructure request. The change owner confirms that support can identify the new behavior in an incident.

The standard release window is Wednesday from 14:00 to 16:00 UTC. Reserve a slot with the release coordinator before publishing a customer notice. A late start requires the coordinator to confirm that enough time remains for a rollback. Do not extend a change into an announced maintenance freeze without the release coordinator's approval.

Roll out to a small audience first when a safe split is available. Compare error and latency trends with the expected range and record the checkpoints in the case. If an alert appears, pause the rollout while the owner determines whether it is related. Revert when the agreed rollback condition is met and verify that the original behavior has returned.

Urgent fixes outside the standard window use the emergency approval path. The owner records why a delay would create more risk than an off-window release. Once the release is complete, update the schedule and confirm the expected state with the duty team. A postponed release keeps the earlier decision in the case and receives a fresh slot.
