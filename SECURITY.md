# Security Policy — Mindforge

## 🔒 Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.x.x   | ✅        |
| < 1.0   | ❌        |

## 🚨 Reporting a Vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.**

Please report via GitHub Security Advisories or email.

### What to include:
- Type of vulnerability
- Source file(s) related to the vulnerability
- Steps to reproduce
- Proof-of-concept code (if possible)
- Impact assessment

## 🛡️ Security Best Practices

- Never hardcode API keys — use environment variables
- Use `.env` files locally (never commit them)
- Run `pre-commit install` to enable secret detection hooks
- Enable 2FA on your GitHub account
- Keep dependencies updated: `pip list --outdated`
- Run `bandit -r .` periodically for static security analysis

## ⏱️ Response Timeline

- Acknowledgement: within 48 hours
- Patch release: within 30 days (critical: within 7 days)
