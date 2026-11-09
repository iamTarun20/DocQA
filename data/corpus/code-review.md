# Code review and protected branches

Northwind Labs engineering projects use protected production branches. A contributor opens a review with a concise explanation of the behavior change, relevant checks and a plan for enabling or rolling back the change. Draft reviews are suitable for design discussion. Move a review out of draft when the implementation can be exercised by another engineer.

A production change needs 2 independent approvals before merge. The author cannot count their own approval. A change that touches a directory with a designated owner must include one approval from that owner group. If the author updates a material part of the diff, reviewers should inspect it again before the merge is recorded.

Reviewers check error handling, maintainability and any effect on stored data. Run or inspect the relevant automated checks, and ask for a reproduction when a behavior is hard to infer from a diff. Separate blocking issues from optional suggestions in review comments. For a change that alters access checks, include a reviewer familiar with the permission model.

An emergency exception is recorded in the incident timeline and approved by the incident commander. Once the emergency is over, open a follow-up review with the tests and any cleanup still due. Branch protection changes are tracked as production changes themselves. Ask a repository owner to verify that review settings have returned to their normal state after an exception.
