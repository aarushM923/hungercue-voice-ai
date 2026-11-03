# Project rules
- This repo is being completely rebuilt to match SPEC.md. The old Flask code is outdated; replace it entirely rather than patching it. Keep only what is still useful (e.g. Google Sheets logging, if it fits the new design).
- This is a LIVE system that real restaurants and shelters call. Work on the v2 branch. Never change the live Twilio number's settings or deploy without asking me first.
- Build the entire spec in one continuous session. Only stop when you need something from me: API keys, accounts, approval to spend money (calls, SMS, paid APIs), or a decision only I can make.
- Python 3.11, type hints, pytest tests for every module. Run all tests before saying you're done.
- Use a virtual environment in .venv. Secrets live in .env, never committed. Free downloads and package installs are fine.
- Commit after each major piece works, with clear messages.
- Never add Claude as a co-author or mention Claude in commit messages or pull requests. In the commit messages don't make it AI-ish, make it clear and avoid symbols.
- I am new to programming tools. Explain each command before running it, and explain non-obvious code in comments. I need to be able to explain every part in an interview.