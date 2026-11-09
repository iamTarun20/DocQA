# Backup and restore standards

Each Northwind Labs production service has an owner who records which databases, object stores and configuration artifacts are covered by scheduled backups. The service inventory names the storage region and recovery contact. Where a service depends on a shared platform, its owner checks whether the platform backup includes the application's own state.

The standard recovery point objective for a production service is 6 hours. The recovery point objective expresses the maximum tolerable age of restored data after an outage. Services with tighter contractual targets record them in their recovery plan. The standard recovery time objective is 12 hours, which describes elapsed time until service restoration.

An operator restores a representative copy in an isolated workspace at least once per quarter. Check schema, access controls and a small set of known records before declaring the exercise complete. The operator saves the start time, completion time and backup timestamp in the exercise log. A restore that opens but cannot answer the representative checks remains unresolved.

Failed scheduled backups create a ticket in the platform duty queue. Fix the schedule and run a fresh backup if the previous good copy will fall outside the service objective. For an actual outage, agree on the recovery point with the incident commander before replacing live state. Record any period of lost writes so the owning team can plan reconciliation with users.
