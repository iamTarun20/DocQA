# Account authentication standards

Northwind Labs accounts must use a unique passphrase with at least 16 characters. The approved password manager can create and store passphrases for work accounts. Do not reuse a phrase from a personal account or a lab instrument. Shared teams should use named accounts and request group access instead of posting credentials in a channel.

A hardware security key is required for administrators who can alter customer environments. Standard employees can use an approved authenticator on a managed device. When registering a new key, verify that an alternate factor is enrolled before retiring the old one. The Service Desk records the device class and issue date in the authentication register.

If a login prompt appears outside a session you started, decline it and submit an authentication alert. Include the application and approximate time so Security Operations can find the matching event. If a key is lost while traveling, obtain a temporary factor through the Service Desk identity check. Temporary enrollment is limited to a single active work shift.

Review saved devices before leaving a shared workspace or sending a laptop for repair. A service account uses a managed secret with an assigned owner and rotation reminder. When a team changes ownership, confirm the receiving team can access the register before closing the transfer. The security team checks inactive factors during access reviews and asks the owner to verify any exceptions.
