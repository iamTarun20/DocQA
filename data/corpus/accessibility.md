# Product accessibility review

Northwind Labs includes an accessibility review before releasing a new customer workflow. The team describes the intended tasks and checks them using keyboard navigation and a screen reader. Review the default state, empty state, validation errors and confirmation screen. A design owner records the checks beside the change so engineering can reproduce the expected interactions.

The standard text resizing check is at 200% zoom. During that check, controls and essential text must remain visible and usable without losing access to the main task. Check dialogs and error summaries as well as the primary form. When a screen reflows, its reading order should still follow the task sequence.

Interactive elements need a visible focus indication and a name that conveys their purpose. Do not rely on color alone to communicate a status or error. If a table has a compact visual layout, verify that its headers still make sense to someone reading one cell at a time. Captions for training videos are reviewed for timing and technical terms.

Log accessibility defects with the affected task, input method and steps to reproduce. A blocker that prevents completing a core task must be fixed or explicitly escalated before release. After changing a shared component, check another screen that uses the component. The design owner records any deferred improvement and names the team responsible for following up.
