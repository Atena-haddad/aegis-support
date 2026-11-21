# Authentication, 2FA & SSO Troubleshooting (SOP-102)
- **SAML Error 403 on Okta SSO**: Indicates the user's email attribute in the IdP assertion does not match the verified workspace domain, or the user is not assigned to the `aegis-prod-users` group in Okta.
- **2FA Token Reset Lockout**: After resetting a TOTP authenticator app, users must use one of their 8-digit backup recovery codes. If lost, workspace admins must submit a verified identity reset request; SLA is 4 business hours.
- **Invalid Bearer Token (HTTP 401)**: Production API keys start with `ags_live_`. Keys rotated in the console remain active for a 15-minute grace period before revocation.
- **Session Expiring Every 5 Minutes**: Triggered when a corporate VPN rotates client IP addresses mid-session while Strict IP Binding is enabled in Workspace Security Settings.
